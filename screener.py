"""
cup_with_handle_screener.py
===========================
カップウィズハンドル 完全自動スクリーニングシステム
ファネル型判定アルゴリズム実装

処理フロー:
  tickers.csv
      │
      ▼
  [Stage 1] ルールベース粗フィルター  (pandas ベクトル演算 / ~4000→100-200銘柄)
      │
      ▼
  [Stage 2] 軽量波形認識フィルター    (NumPy 配列スライス / →30-50銘柄)
      │
      ▼
  [Stage 3] LightGBM 勝率予測        (model.pkl ロード済みモデルで推論)
      │
      ▼
  利確・損切りライン算出
      │
      ▼
  Supabase TRUNCATE & INSERT
      │
      ▼
  LINE / Discord 通知
"""
from dotenv import load_dotenv
load_dotenv()  
import os
import time
import logging
import pickle
import warnings
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta          # TA-Lib の代替 (pip install pandas-ta)
import yfinance as yf
from supabase import create_client, Client

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 定数・設定
# ─────────────────────────────────────────────
TICKERS_CSV         = "tickers.csv"          # ticker列を持つCSVファイル
MODEL_PATH          = "model.pkl"            # 学習済み LightGBM モデル
BATCH_SIZE          = 50                     # yfinance 1回のダウンロード銘柄数
SLEEP_BETWEEN_BATCH = 2.0                    # バッチ間のsleep秒数
MAX_RETRY           = 3                      # yfinanceリトライ回数
HISTORY_DAYS        = 400                    # 取得する履歴日数 (200日MA算出に400日必要)

# カップウィズハンドルの成立条件定数
CUP_MIN_WEEKS       = 7                      # カップ最小期間(週)
CUP_MAX_WEEKS       = 65                     # カップ最大期間(週)
CUP_MIN_DEPTH_PCT   = 12.0                   # カップ最小深さ(%)
CUP_MAX_DEPTH_PCT   = 33.0                   # カップ最大深さ(%)
HANDLE_MAX_DEPTH_PCT = 12.0                  # ハンドル最大深さ(%)
HANDLE_MIN_DAYS     = 5                      # ハンドル最小期間(日)
HANDLE_MAX_DAYS     = 14                     # ハンドル最大期間(日)
PRIOR_UPTREND_MIN_PCT = 30.0                 # カップ前上昇率の最低値(%)
BREAKOUT_VOL_RATIO_MIN = 1.4                 # ブレイク時出来高/平均の最低倍率

# LightGBM 閾値
LGBM_THRESHOLD      = 0.55                   # 勝率がこれ以上の銘柄を最終候補とする

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# データ取得レイヤー
# ─────────────────────────────────────────────

def load_tickers(path: str = TICKERS_CSV) -> list[str]:
    """tickers.csv から銘柄リストを読み込む"""
    df = pd.read_csv(path)
    # 列名が "ticker" or "Ticker" or "symbol" 等に対応
    col = [c for c in df.columns if c.lower() in ("ticker", "symbol")][0]
    tickers = df[col].dropna().str.upper().tolist()
    log.info(f"Loaded {len(tickers)} tickers from {path}")
    return tickers


def download_history(tickers: list[str], days: int = HISTORY_DAYS) -> dict[str, pd.DataFrame]:
    """
    yfinance でバッチ処理 + リトライ付きの履歴ダウンロード。
    Returns: {ticker: DataFrame(columns=[Open,High,Low,Close,Volume])}
    """
    end   = date.today()
    start = end - timedelta(days=days)
    result: dict[str, pd.DataFrame] = {}

    # BATCH_SIZE 銘柄ずつチャンク分割
    chunks = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    log.info(f"Downloading {len(tickers)} tickers in {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks, 1):
        for attempt in range(1, MAX_RETRY + 1):
            try:
                raw = yf.download(
                    chunk,
                    start=str(start),
                    end=str(end),
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                # 1銘柄の場合 MultiIndex にならないケース対応
                if len(chunk) == 1:
                    df = raw.copy()
                    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
                    if not df.empty:
                        result[chunk[0]] = df
                else:
                    for ticker in chunk:
                        try:
                            df = raw[ticker].dropna(how="all")
                            if len(df) > 50:
                                result[ticker] = df
                        except KeyError:
                            pass
                break  # 成功したらリトライループを抜ける
            except Exception as e:
                log.warning(f"Chunk {i} attempt {attempt} failed: {e}")
                if attempt < MAX_RETRY:
                    time.sleep(5 * attempt)
                else:
                    log.error(f"Chunk {i} skipped after {MAX_RETRY} retries.")

        time.sleep(SLEEP_BETWEEN_BATCH)
        log.info(f"  Chunk {i}/{len(chunks)} done ({len(result)} tickers so far)")

    log.info(f"Download complete: {len(result)} tickers with valid data")
    return result


# ─────────────────────────────────────────────
# Stage 1: ルールベース粗フィルター
# ─────────────────────────────────────────────

def stage1_rule_filter(all_data: dict[str, pd.DataFrame]) -> list[str]:
    """
    pandas のベクトル演算で全銘柄を一括スクリーニング。
    重い処理の前に ~4000 → 100-200 銘柄に絞り込む。

    成立条件:
      - 現在値 > 200日移動平均線 (上昇トレンド確認)
      - 現在値が 52週高値の 75%以上 (高値圏にいる)
      - 平均出来高 > 100,000株 (流動性フィルター)
      - 株価 > $10 (ペニー株除外)
    """
    passed = []

    for ticker, df in all_data.items():
        try:
            close  = df["Close"]
            volume = df["Volume"]
            if len(close) < 210:          # データ不足は除外
                continue

            # 200日移動平均 (pandas_ta を使わず自前で計算: 軽量)
            ma200  = close.rolling(200).mean()
            ma50   = close.rolling(50).mean()

            current   = close.iloc[-1]
            ma200_now = ma200.iloc[-1]
            ma50_now  = ma50.iloc[-1]

            # ── 条件1: 現在値 > 200日MA ──
            if current <= ma200_now:
                continue

            # ── 条件2: 52週高値の 75%以上 ──
            high_52w = close.iloc[-252:].max() if len(close) >= 252 else close.max()
            if current < high_52w * 0.75:
                continue

            # ── 条件3: 平均出来高 > 100,000 ──
            avg_vol = volume.iloc[-60:].mean()
            if avg_vol < 100_000:
                continue

            # ── 条件4: 株価 > $10 ──
            if current < 10.0:
                continue

            # ── 条件5: 50日MA > 200日MA (ゴールデンクロス状態) ──
            if ma50_now <= ma200_now:
                continue

            passed.append(ticker)

        except Exception:
            continue

    log.info(f"Stage 1 passed: {len(passed)} / {len(all_data)}")
    return passed


# ─────────────────────────────────────────────
# Stage 2: 軽量波形認識フィルター ← 最重要実装
# ─────────────────────────────────────────────

def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Min-Max正規化: 配列全体を 0.0~1.0 にスケーリング"""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def _detect_cup_shape(
    norm_close: np.ndarray,   # Min-Max正規化済み終値配列 (長さ = cup_window)
) -> dict | None:
    """
    U字カップ形状を「配列スライスと条件分岐」で検知する軽量ロジック。
    DTW・機械学習は使わない。

    アルゴリズム:
      1. 配列を左1/3・中1/3・右1/3 に3分割
      2. 各セグメントの平均値を比較
         - 左 > 中 (左肩が底より高い)
         - 右 > 中 (右肩が底より高い)
         - 左 ≈ 右 (対称性: 左右の差が閾値以内)
      3. 中央セグメントの「底の丸さ」を評価
         - 底付近 (下位20%の値) の占める割合が少なすぎない (V字でない)
         - 底部分の標準偏差が小さい (なだらかなU字)
    """
    n = len(norm_close)
    if n < 30:
        return None

    # ── セグメント分割 ──
    seg = n // 3
    left_seg   = norm_close[:seg]           # 左1/3
    mid_seg    = norm_close[seg:2*seg]      # 中1/3
    right_seg  = norm_close[2*seg:]         # 右1/3

    left_mean  = left_seg.mean()
    mid_mean   = mid_seg.mean()
    right_mean = right_seg.mean()

    # ── 条件A: 左肩・右肩が底より高い ──
    # 閾値0.15: 正規化値で0.15以上の差が必要 (ノイズ排除)
    if left_mean - mid_mean < 0.15:
        return None
    if right_mean - mid_mean < 0.15:
        return None

    # ── 条件B: 対称性チェック ──
    # 左右の平均値の差が 0.20 以内 (完全一致不要だが大幅な非対称はNG)
    symmetry_score = 1.0 - abs(left_mean - right_mean)  # 1に近いほど対称
    if abs(left_mean - right_mean) > 0.20:
        return None

    # ── 条件C: V字でなくU字であること ──
    # 底付近 (正規化値が 0.25 以下) の連続日数を確認
    # 連続5日以上が底付近にある = ゆるやかなU字
    bottom_mask       = mid_seg < 0.25          # 底付近の定義
    bottom_run_length = _max_consecutive_true(bottom_mask)
    if bottom_run_length < 5:
        # V字: 底が一点で即反発している
        return None

    # 底セグメントの標準偏差が小さい (なだらか)
    bottom_std = mid_seg[bottom_mask].std() if bottom_mask.sum() > 2 else 1.0
    if bottom_std > 0.15:
        return None

    # ── 右肩が左肩の90%以上 ──
    # (右肩が低すぎるカップは除外: 完全なU字の右半分が必要)
    if right_mean < left_mean * 0.85:
        return None

    return {
        "left_mean":      float(left_mean),
        "mid_mean":       float(mid_mean),
        "right_mean":     float(right_mean),
        "symmetry_score": float(symmetry_score),
        "bottom_run":     int(bottom_run_length),
    }


def _max_consecutive_true(mask: np.ndarray) -> int:
    """Boolean配列でTrueが連続する最大長を返す"""
    max_run = current_run = 0
    for v in mask:
        if v:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _detect_handle_shape(
    norm_close: np.ndarray,    # Min-Max正規化済み (ハンドル候補期間)
    norm_volume: np.ndarray,   # Min-Max正規化済み出来高
) -> dict | None:
    """
    ハンドル形状を検知する軽量ロジック。

    成立条件:
      1. ハンドルの高値がカップ右肩の上半分にある (norm_close先頭 > 0.5)
      2. 全体として緩やかな下落トレンド (線形回帰の傾きが負)
      3. 深さが 12% 以内 (raw price での深さは呼び出し元でチェック)
      4. 出来高が減少傾向 (線形回帰の傾きが負)
    """
    n = len(norm_close)
    if n < HANDLE_MIN_DAYS or n > HANDLE_MAX_DAYS * 3:
        return None

    # ── 条件1: ハンドルはカップ右肩の上半分で形成 ──
    # 正規化価格の先頭値 (ハンドル開始時) が全体の 0.5 以上
    if norm_close[0] < 0.5:
        return None

    # ── 条件2: ハンドル全体が緩やかな下落トレンド ──
    # 線形回帰の傾き (numpy の polyfit)
    x = np.arange(n, dtype=float)
    price_slope = np.polyfit(x, norm_close, 1)[0]   # 傾き (1次項係数)
    if price_slope >= 0:
        # 下落していない (横ばいor上昇) はハンドルではない
        return None

    # ── 条件3: ハンドル内の値動きが小さい ──
    # 高値と安値の差が正規化空間で 0.35 以内
    handle_range = norm_close.max() - norm_close.min()
    if handle_range > 0.35:
        return None

    # ── 条件4: 出来高が減少傾向 ──
    vol_slope = np.polyfit(x, norm_volume, 1)[0]
    volume_declining = (vol_slope < 0)

    # 出来高減少スコア (0~1): 傾きが負で急なほど1に近い
    vol_decline_score = min(1.0, abs(vol_slope) * 10) if volume_declining else 0.0

    return {
        "price_slope":       float(price_slope),
        "handle_range":      float(handle_range),
        "volume_declining":  bool(volume_declining),
        "vol_decline_score": float(vol_decline_score),
    }


def stage2_shape_filter(
    tickers: list[str],
    all_data: dict[str, pd.DataFrame],
) -> list[dict]:
    """
    第2段階: 軽量波形認識でカップとハンドルの形状を検知する。

    Returns:
        各銘柄の形状パラメータを含む dict のリスト
    """
    passed = []

    for ticker in tickers:
        df = all_data.get(ticker)
        if df is None or len(df) < 100:
            continue

        close  = df["Close"].values.astype(float)
        volume = df["Volume"].values.astype(float)
        dates  = df.index

        # ── カップ候補ウィンドウのスキャン ──
        # CUP_MIN_WEEKS~CUP_MAX_WEEKS の範囲で直近データを探索
        # 計算コスト節約のため、直近200日だけを対象とする

        best_result = None
        best_score  = -1.0

        # カップ長さを週単位で試す (7週, 14週, ..., 65週 → 営業日換算)
        for cup_weeks in range(CUP_MIN_WEEKS, min(CUP_MAX_WEEKS, 40), 4):
            cup_days = cup_weeks * 5   # 週 → 営業日 (概算)
            if cup_days > len(close) - 15:
                break

            # カップ候補区間: 末尾から cup_days + handle期間 前まで
            handle_days = HANDLE_MAX_DAYS
            cup_end_idx = len(close) - handle_days       # カップ右肩位置
            cup_start_idx = cup_end_idx - cup_days

            if cup_start_idx < 0:
                continue

            cup_close  = close[cup_start_idx:cup_end_idx]
            cup_volume = volume[cup_start_idx:cup_end_idx]

            # ── カップ深さ検証 (raw price) ──
            cup_high_left  = cup_close[:len(cup_close)//4].max()   # 左肩高値
            cup_high_right = cup_close[-len(cup_close)//4:].max()  # 右肩高値
            cup_bottom_val = cup_close.min()

            # 左右両方の肩を基準に深さを計算
            ref_high   = max(cup_high_left, cup_high_right)
            depth_pct  = (ref_high - cup_bottom_val) / ref_high * 100

            if not (CUP_MIN_DEPTH_PCT <= depth_pct <= CUP_MAX_DEPTH_PCT):
                continue

            # ── カップ前の上昇トレンド検証 ──
            # カップ開始前120日間の安値→カップ左肩高値の上昇率
            prior_start = max(0, cup_start_idx - 120)
            prior_close = close[prior_start:cup_start_idx]
            if len(prior_close) < 20:
                continue
            prior_low   = prior_close.min()
            prior_uptrend_pct = (cup_high_left - prior_low) / prior_low * 100
            if prior_uptrend_pct < PRIOR_UPTREND_MIN_PCT:
                continue

            # ── Min-Max正規化してカップ形状を判定 ──
            norm_cup_close = _normalize_minmax(cup_close)
            cup_result = _detect_cup_shape(norm_cup_close)
            if cup_result is None:
                continue

            # ── ハンドル候補区間 ──
            handle_close  = close[cup_end_idx:]
            handle_volume = volume[cup_end_idx:]

            if len(handle_close) < HANDLE_MIN_DAYS:
                continue

            # ハンドル深さ検証 (raw price)
            handle_high_val  = handle_close.max()
            handle_low_val   = handle_close.min()
            handle_depth_pct = (handle_high_val - handle_low_val) / handle_high_val * 100

            if handle_depth_pct > HANDLE_MAX_DEPTH_PCT:
                continue

            # ── ハンドル正規化 + 形状判定 ──
            norm_handle_close  = _normalize_minmax(handle_close)
            norm_handle_volume = _normalize_minmax(handle_volume)
            handle_result = _detect_handle_shape(norm_handle_close, norm_handle_volume)
            if handle_result is None:
                continue

            # ── ブレイクアウト判定 ──
            # 現在値がハンドル高値を超えているか
            current_price   = close[-1]
            breakout_price  = handle_high_val
            is_breaking_out = current_price >= breakout_price

            # 出来高急増チェック
            avg_vol_20d     = volume[-20:].mean() if len(volume) >= 20 else volume.mean()
            today_vol       = volume[-1]
            breakout_vol_ratio = today_vol / avg_vol_20d if avg_vol_20d > 0 else 0.0

            # ブレイクアウト中は出来高急増が必要
            if is_breaking_out and breakout_vol_ratio < BREAKOUT_VOL_RATIO_MIN:
                continue

            # ── 200日MAとの乖離 ──
            ma200_arr = pd.Series(close).rolling(200).mean().values
            ma200_now = ma200_arr[-1]
            price_vs_ma200_pct = (current_price - ma200_now) / ma200_now * 100

            # ── スコア計算 (総合評価指標) ──
            score = (
                cup_result["symmetry_score"] * 0.4 +   # U字対称性 (重要)
                handle_result["vol_decline_score"] * 0.3 +  # ハンドル中の出来高減少
                min(breakout_vol_ratio / 5.0, 1.0) * 0.3   # ブレイク時出来高 (最大5倍でフル)
            )

            if score > best_score:
                best_score = score
                best_result = {
                    "ticker":              ticker,
                    "current_price":       float(current_price),
                    # カップパラメータ
                    "cup_start_date":      dates[cup_start_idx].date(),
                    "cup_end_date":        dates[cup_end_idx - 1].date(),
                    "cup_left_high":       float(cup_high_left),
                    "cup_right_high":      float(cup_high_right),
                    "cup_bottom":          float(cup_bottom_val),
                    "cup_depth_pct":       float(depth_pct),
                    "cup_duration_weeks":  cup_weeks,
                    "cup_symmetry_score":  float(cup_result["symmetry_score"]),
                    # ハンドルパラメータ
                    "handle_start_date":   dates[cup_end_idx].date(),
                    "handle_end_date":     dates[-1].date(),
                    "handle_high":         float(handle_high_val),
                    "handle_low":          float(handle_low_val),
                    "handle_depth_pct":    float(handle_depth_pct),
                    "handle_duration_days": int(len(handle_close)),
                    "handle_volume_decline_pct": float(handle_result["vol_decline_score"] * 100),
                    # ブレイクアウト
                    "breakout_price":      float(breakout_price),
                    "is_breaking_out":     bool(is_breaking_out),
                    "breakout_volume_ratio": float(breakout_vol_ratio),
                    # トレンド
                    "prior_uptrend_pct":   float(prior_uptrend_pct),
                    "price_vs_ma200_pct":  float(price_vs_ma200_pct),
                    # スコア
                    "shape_score":         float(score),
                    # 生データ (特徴量生成用)
                    "_close":              close,
                    "_volume":             volume,
                    "_dates":             dates,
                }

        if best_result is not None:
            passed.append(best_result)

    log.info(f"Stage 2 passed: {len(passed)} / {len(tickers)}")
    return passed


# ─────────────────────────────────────────────
# Stage 3: LightGBM 勝率予測
# ─────────────────────────────────────────────

def _build_features(rec: dict) -> dict:
    """
    LightGBM に渡す特徴量ベクトルを構築する。
    特徴量選定の根拠:
      - Volume Spike: ブレイク時の出来高急増は成功シグナル
      - Volatility Contraction: ハンドル中のボラ縮小はエネルギー蓄積を示す
      - RS Rank: 相対強度 (市場全体との比較) が高い銘柄が有利
      - Cup Symmetry: 対称性が高いほど機関投資家の売り圧力が解消済み
    """
    close  = rec["_close"]
    volume = rec["_volume"]

    # ── 出来高スパイク比率 ──
    # 直近5日の出来高平均 / 過去60日の出来高平均
    avg_vol_60d = volume[-60:].mean() if len(volume) >= 60 else volume.mean()
    avg_vol_5d  = volume[-5:].mean()  if len(volume) >= 5  else volume.mean()
    vol_spike_ratio = avg_vol_5d / avg_vol_60d if avg_vol_60d > 0 else 1.0

    # ── ボラティリティ収縮率 ──
    # ハンドル期間のATR / カップ期間のATR
    # pandas_ta で ATR 計算 (TA-Lib 不使用)
    df_tmp = pd.DataFrame({
        "High":  close,   # 簡易版: OHLC がなければ Close で代替
        "Low":   close,
        "Close": close,
    })
    atr_series = ta.atr(df_tmp["High"], df_tmp["Low"], df_tmp["Close"], length=14)
    if atr_series is not None and len(atr_series) > 30:
        handle_len   = rec.get("handle_duration_days", 10)
        atr_handle   = atr_series.iloc[-handle_len:].mean()
        atr_cup      = atr_series.iloc[-handle_len-rec["cup_duration_weeks"]*5:-handle_len].mean()
        volatility_contraction = atr_handle / atr_cup if atr_cup > 0 else 1.0
    else:
        volatility_contraction = 1.0

    # ── 相対強度ランク (RS Rank) ──
    # 52週パフォーマンス (単純: 現在値 / 52週前の値 - 1)
    period_252 = close[-252:] if len(close) >= 252 else close
    rs_raw = close[-1] / period_252[0] - 1 if period_252[0] > 0 else 0.0

    # ── 価格モメンタム ──
    # 直近20日のリターン
    momentum_20d = (close[-1] / close[-20] - 1) if len(close) >= 20 else 0.0

    # ── カップのU字スコア ──
    cup_symmetry = rec.get("cup_symmetry_score", 0.5)
    cup_depth    = rec.get("cup_depth_pct", 20.0) / 100.0

    # ── 200日MAからの乖離率 ──
    ma200_pct = rec.get("price_vs_ma200_pct", 0.0) / 100.0

    # ── ハンドル深さ ──
    handle_depth = rec.get("handle_depth_pct", 8.0) / 100.0

    # ── ブレイクアウト出来高比率 ──
    breakout_vol = rec.get("breakout_volume_ratio", 1.0)

    features = {
        "vol_spike_ratio":         float(vol_spike_ratio),
        "volatility_contraction":  float(volatility_contraction),
        "rs_rank":                 float(rs_raw),
        "momentum_20d":            float(momentum_20d),
        "cup_symmetry_score":      float(cup_symmetry),
        "cup_depth_pct":           float(cup_depth),
        "handle_depth_pct":        float(handle_depth),
        "breakout_vol_ratio":      float(breakout_vol),
        "price_vs_ma200_pct":      float(ma200_pct),
        "prior_uptrend_pct":       float(rec.get("prior_uptrend_pct", 0.0) / 100.0),
        "cup_duration_weeks":      float(rec.get("cup_duration_weeks", 10)),
        "handle_duration_days":    float(rec.get("handle_duration_days", 7)),
    }
    return features


def stage3_lgbm_filter(candidates: list[dict]) -> list[dict]:
    """
    第3段階: 学習済み LightGBM モデルで勝率を予測し閾値でフィルタリング。
    model.pkl が存在しない場合は shape_score でフォールバック。
    """
    # モデルロード
    model = None
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        log.info(f"LightGBM model loaded from {MODEL_PATH}")
    else:
        log.warning(f"{MODEL_PATH} not found. Using shape_score as fallback.")

    passed = []
    feature_cols = None   # 最初の1件で列順を確定

    for rec in candidates:
        feat_dict = _build_features(rec)
        rec["features_json"] = feat_dict   # Supabaseへの保存用

        if model is not None:
            # 列順を model の feature_name_ に揃える
            if feature_cols is None:
                feature_cols = model.feature_name_
            feat_df = pd.DataFrame([feat_dict])[feature_cols]
            prob = float(model.predict_proba(feat_df)[0][1])   # クラス1 (勝ち) の確率
            rec["lgbm_win_prob"] = prob
            rec["funnel_stage"]  = 3

            if prob < LGBM_THRESHOLD:
                continue
        else:
            # フォールバック: shape_score を疑似確率として使用
            rec["lgbm_win_prob"] = float(rec.get("shape_score", 0.5))
            rec["funnel_stage"]  = 2

        passed.append(rec)

    log.info(f"Stage 3 passed: {len(passed)} / {len(candidates)}")
    return passed


# ─────────────────────────────────────────────
# 利確・損切りライン算出
# ─────────────────────────────────────────────

def _find_recent_resistance(close: np.ndarray, lookback: int = 252) -> float:
    """
    過去 lookback 日のチャートから「直近の目立つ高値 (レジスタンス)」を抽出。
    アルゴリズム:
      - ローリング最大値 (20日窓) でピークを特定
      - 複数ピークのうち最も現在値に近い (現在値より上の) ものを選択
    """
    arr = close[-lookback:] if len(close) >= lookback else close
    n   = len(arr)
    if n < 40:
        return float(arr.max())

    # 20日ローリング最大値でピーク検出
    s = pd.Series(arr)
    rolling_max = s.rolling(20, center=True).max()
    is_peak = (s == rolling_max) & (s > s.shift(10)) & (s > s.shift(-10))
    peaks   = s[is_peak].values

    current = close[-1]
    # 現在値より上にあるピークを選ぶ (レジスタンス)
    resistances = peaks[peaks > current]

    if len(resistances) == 0:
        # 全ピークが現在値以下なら最高値を使う
        return float(peaks.max()) if len(peaks) > 0 else float(arr.max())

    # 最も低い (現在値に近い) レジスタンス
    return float(resistances.min())


def calculate_price_targets(rec: dict) -> dict:
    """
    利確A/B・損切りラインを算出してレコードに追加する。

    利確A: ハンドル底 + カップの深さ (値幅) ← 最も一般的なCWHのターゲット
    利確B: 直近レジスタンスの少し下 (2%引き)
    損切り: ハンドル安値の 1% 下 (多少のノイズ許容)
    """
    cup_depth_value  = rec["cup_left_high"] - rec["cup_bottom"]   # カップの絶対値幅
    handle_low       = rec["handle_low"]
    handle_high      = rec["handle_high"]
    close_arr        = rec["_close"]

    # ── 利確ライン A ──
    # ハンドルの底値 + カップの値幅
    take_profit_a = handle_low + cup_depth_value

    # ── 利確ライン B ──
    # 直近レジスタンスの 2% 手前 (到達しにくい高値を避ける)
    resistance = _find_recent_resistance(close_arr)
    take_profit_b = resistance * 0.98

    # 利確Bが利確Aより低い場合は利確Aを優先
    if take_profit_b < take_profit_a:
        take_profit_b = take_profit_a * 1.1

    # ── 損切りライン ──
    # ハンドル安値の 1% 下
    stop_loss = handle_low * 0.99

    return {
        "take_profit_a": float(take_profit_a),
        "take_profit_b": float(take_profit_b),
        "stop_loss":     float(stop_loss),
    }


# ─────────────────────────────────────────────
# OHLCV JSON生成 (Lightweight Charts用)
# ─────────────────────────────────────────────

def build_ohlcv_json(df: pd.DataFrame, days: int = 180) -> list[dict]:
    """
    TradingView Lightweight Charts に渡す OHLCV の JSON 配列を生成。
    フォーマット: [{"time": "2024-01-05", "open": 150.1, ...}, ...]
    """
    tail = df.iloc[-days:] if len(df) >= days else df
    records = []
    for idx, row in tail.iterrows():
        records.append({
            "time":   str(idx.date()),
            "open":   round(float(row["Open"]),   4),
            "high":   round(float(row["High"]),   4),
            "low":    round(float(row["Low"]),    4),
            "close":  round(float(row["Close"]),  4),
            "volume": int(row["Volume"]),
        })
    return records


# ─────────────────────────────────────────────
# Supabase 連携
# ─────────────────────────────────────────────

def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")  # service_role キーを使用
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not set.")
    return create_client(url, key)


def save_to_supabase(
    supabase: Client,
    final_candidates: list[dict],
    all_data: dict[str, pd.DataFrame],
    batch_date: date,
) -> None:
    """
    Supabase の candidates テーブルを TRUNCATE & INSERT で上書き。
    ステートレス運用: 前日のデータは完全削除してから今日の候補を挿入。
    """
    table = supabase.table("candidates")

    # ── TRUNCATE: 前回バッチのデータをすべて削除 ──
    # Supabase REST API に TRUNCATE はないため、バッチ日全削除で代替
    # (本番では Supabase Edge Function や pg_rpc で TRUNCATE を実行推奨)
    log.info("Deleting old candidate data...")
    supabase.table("candidates").delete().neq("id", 0).execute()

    if not final_candidates:
        log.info("No candidates to insert.")
        return

    # ── INSERT ──
    rows = []
    for rec in final_candidates:
        ticker  = rec["ticker"]
        df      = all_data.get(ticker)
        targets = calculate_price_targets(rec)
        ohlcv   = build_ohlcv_json(df) if df is not None else []

        # _で始まる内部用フィールド (numpy配列) を除外
        row = {k: v for k, v in rec.items() if not k.startswith("_")}
        row.update({
            "batch_date":     str(batch_date),
            "screened_at":    datetime.utcnow().isoformat(),
            "take_profit_a":  targets["take_profit_a"],
            "take_profit_b":  targets["take_profit_b"],
            "stop_loss":      targets["stop_loss"],
            "ohlcv_json":     ohlcv,
            # date型をstr変換 (JSON シリアライズ対応)
            "cup_start_date":    str(rec.get("cup_start_date", "")),
            "cup_end_date":      str(rec.get("cup_end_date", "")),
            "handle_start_date": str(rec.get("handle_start_date", "")),
            "handle_end_date":   str(rec.get("handle_end_date", "")),
        })
        # スコアや内部フィールドを除去
        for k in ["shape_score"]:
            row.pop(k, None)

        rows.append(row)

    log.info(f"Inserting {len(rows)} candidates into Supabase...")
    # 100件ずつバッチインサート (RLS / サイズ制限対策)
    for i in range(0, len(rows), 100):
        supabase.table("candidates").insert(rows[i:i+100]).execute()

    log.info("Supabase insert complete.")


# ─────────────────────────────────────────────
# 通知 (LINE / Discord)
# ─────────────────────────────────────────────

def notify_discord(final_candidates: list[dict]) -> None:
    """Discord Webhook でスクリーニング結果を通知する"""
    import requests

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        log.warning("DISCORD_WEBHOOK_URL not set. Skipping notification.")
        return

    if not final_candidates:
        message = "📊 本日のカップウィズハンドル候補: **0銘柄**"
    else:
        lines = ["📊 **カップウィズハンドル候補銘柄**\n"]
        for rec in sorted(final_candidates, key=lambda r: r.get("lgbm_win_prob", 0), reverse=True)[:10]:
            lines.append(
                f"**{rec['ticker']}** | 現在値: ${rec['current_price']:.2f} | "
                f"ブレイクアウト: ${rec['breakout_price']:.2f} | "
                f"勝率予測: {rec.get('lgbm_win_prob', 0):.1%} | "
                f"損切り: ${rec.get('stop_loss', 0):.2f}"
            )
        message = "\n".join(lines)

    payload = {"content": message[:2000]}  # Discord の文字数制限
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        log.info("Discord notification sent.")
    except Exception as e:
        log.error(f"Discord notification failed: {e}")


# ─────────────────────────────────────────────
# メイン実行
# ─────────────────────────────────────────────

def main() -> None:
    batch_date = date.today()
    log.info(f"=== Cup with Handle Screener: {batch_date} ===")

    supabase = get_supabase_client()

    # バッチ開始ログを記録（安全に挿入→衝突時は更新）
    started_at = datetime.utcnow().isoformat()
    try:
        supabase.table("batch_logs").insert({
            "batch_date": str(batch_date),
            "started_at": started_at,
            "status":     "running",
        }).execute()
    except Exception as e:
        # 既に同日付の行がある（重複キー）の場合は更新にフォールバック
        if "duplicate key" in str(e) or "23505" in str(e):
            supabase.table("batch_logs").update({
                "started_at": started_at,
                "status":     "running",
            }).eq("batch_date", str(batch_date)).execute()
        else:
            raise

    try:
        # ── データ取得 ──
        # tickers.csv の存在確認
        if not os.path.exists(TICKERS_CSV):
            log.error(f"{TICKERS_CSV} not found; aborting batch.")
            supabase.table("batch_logs").update({
                "finished_at": datetime.utcnow().isoformat(),
                "status": "failed",
                "error_message": f"{TICKERS_CSV} not found",
            }).eq("batch_date", str(batch_date)).execute()
            return

        tickers  = load_tickers(TICKERS_CSV)
        all_data = download_history(tickers)

        # ── Stage 1: ルールベース粗フィルター ──
        s1_passed = stage1_rule_filter(all_data)

        # ── Stage 2: 形状フィルター ──
        s2_passed = stage2_shape_filter(s1_passed, all_data)

        # ── Stage 3: LightGBM 勝率予測 ──
        s3_passed = stage3_lgbm_filter(s2_passed)

        # ── Supabase 保存 ──
        save_to_supabase(supabase, s3_passed, all_data, batch_date)

        # ── 通知 ──
        notify_discord(s3_passed)

        # バッチ完了ログ: 既存行を更新して上書き
        supabase.table("batch_logs").update({
            "finished_at":   datetime.utcnow().isoformat(),
            "status":        "success",
            "total_tickers": len(tickers),
            "stage1_passed": len(s1_passed),
            "stage2_passed": len(s2_passed),
            "stage3_passed": len(s3_passed),
        }).eq("batch_date", str(batch_date)).execute()

        log.info(f"=== Batch complete. Final candidates: {len(s3_passed)} ===")

    except Exception as e:
        log.exception(f"Batch failed: {e}")
        supabase.table("batch_logs").update({
            "finished_at":   datetime.utcnow().isoformat(),
            "status":        "failed",
            "error_message": str(e),
        }).eq("batch_date", str(batch_date)).execute()
        raise


if __name__ == "__main__":
    main()
