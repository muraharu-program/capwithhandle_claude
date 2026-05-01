-- =============================================================
-- Cup with Handle スクリーナー: Supabase スキーマ定義
-- =============================================================
-- 運用方針: 日次バッチで TRUNCATE & INSERT (全量上書き)
-- PythonからのInsertとNext.jsからのFetchの契約となる定義
-- =============================================================


-- -----------------------------------------------------------
-- 1. candidates テーブル
--    スクリーニングを通過した候補銘柄の最終結果を格納
--    日次バッチ完了時に TRUNCATE → INSERT で全量上書き
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS candidates (
    -- 主キー (バッチ実行日 + ティッカー)
    id                      BIGSERIAL PRIMARY KEY,
    ticker                  TEXT        NOT NULL,
    company_name            TEXT,
    screened_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),   -- バッチ実行日時
    batch_date              DATE        NOT NULL,                  -- バッチ対象日 (JST)

    -- ---- 現在値・基本情報 ----
    current_price           NUMERIC(12, 4) NOT NULL,
    volume_today            BIGINT,
    market_cap              BIGINT,
    sector                  TEXT,
    industry                TEXT,

    -- ---- カップ形状パラメータ ----
    cup_start_date          DATE,        -- カップ左肩の日付
    cup_end_date            DATE,        -- カップ右肩の日付
    cup_left_high           NUMERIC(12, 4),   -- 左肩の高値
    cup_right_high          NUMERIC(12, 4),   -- 右肩の高値
    cup_bottom              NUMERIC(12, 4),   -- カップ底値
    cup_depth_pct           NUMERIC(6, 2),    -- カップの深さ (%)
    cup_duration_weeks      INTEGER,          -- カップの期間 (週数)
    cup_symmetry_score      NUMERIC(6, 4),    -- U字対称性スコア (0~1)

    -- ---- ハンドル形状パラメータ ----
    handle_start_date       DATE,
    handle_end_date         DATE,
    handle_high             NUMERIC(12, 4),   -- ハンドル高値 (ブレイクアウトライン)
    handle_low              NUMERIC(12, 4),   -- ハンドル安値
    handle_depth_pct        NUMERIC(6, 2),    -- ハンドルの深さ (%)
    handle_duration_days    INTEGER,
    handle_volume_decline_pct NUMERIC(6, 2),  -- ハンドル中の出来高減少率 (%)

    -- ---- ブレイクアウト判定 ----
    breakout_price          NUMERIC(12, 4),   -- ブレイクアウトライン (= handle_high)
    is_breaking_out         BOOLEAN   NOT NULL DEFAULT FALSE,  -- 当日ブレイクアウト中か
    breakout_volume_ratio   NUMERIC(6, 2),    -- ブレイク時出来高/20日平均出来高

    -- ---- エントリー前トレンド ----
    prior_uptrend_pct       NUMERIC(6, 2),    -- カップ形成前の上昇率 (%)
    price_vs_ma200_pct      NUMERIC(6, 2),    -- 現在値が200日MAから何%上か

    -- ---- 利確・損切りライン ----
    take_profit_a           NUMERIC(12, 4),   -- 利確A: ハンドル底 + カップ深さ
    take_profit_b           NUMERIC(12, 4),   -- 利確B: 直近目立つ高値の少し下
    stop_loss               NUMERIC(12, 4),   -- 損切り: ハンドル安値を割った価格

    -- ---- ML スコア ----
    lgbm_win_prob           NUMERIC(6, 4),    -- LightGBM 勝率予測 (0.0~1.0)
    funnel_stage            INTEGER NOT NULL, -- 通過した最終ファネル段階 (1/2/3)

    -- ---- 特徴量スナップショット (JSON) ----
    -- LightGBMに渡した特徴量の生ベクトルを保存 (デバッグ・モデル改善用)
    features_json           JSONB,

    -- ---- チャートデータ (Lightweight Charts用) ----
    -- 過去180日のOHLCVをJSON配列で保存 (フロントからの追加フェッチを不要にする)
    ohlcv_json              JSONB,

    -- ユニーク制約: 同一バッチ日に同一ティッカーは1行のみ
    CONSTRAINT uq_candidates_batch_ticker UNIQUE (batch_date, ticker)
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_candidates_batch_date   ON candidates(batch_date DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_lgbm_score   ON candidates(lgbm_win_prob DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_breakout     ON candidates(is_breaking_out) WHERE is_breaking_out = TRUE;
CREATE INDEX IF NOT EXISTS idx_candidates_ticker       ON candidates(ticker);

COMMENT ON TABLE candidates IS
    '日次バッチで全量上書きされるカップウィズハンドル候補銘柄テーブル。'
    'batch_date が最新のレコード群が当日の候補リスト。';

COMMENT ON COLUMN candidates.ohlcv_json IS
    '例: [{"time":"2024-01-05","open":150.1,"high":152.3,"low":149.8,"close":151.5,"volume":1234567}, ...]';

COMMENT ON COLUMN candidates.features_json IS
    'LightGBMへの入力特徴量スナップショット。'
    '例: {"vol_spike_ratio":2.3,"volatility_contraction":0.42,"rs_rank":87,...}';


-- -----------------------------------------------------------
-- 2. batch_logs テーブル
--    バッチ実行の履歴・統計を記録 (上書きせず累積)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS batch_logs (
    id                  BIGSERIAL PRIMARY KEY,
    batch_date          DATE        NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL,
    finished_at         TIMESTAMPTZ,
    status              TEXT        NOT NULL DEFAULT 'running',  -- running / success / failed
    total_tickers       INTEGER,
    stage1_passed       INTEGER,
    stage2_passed       INTEGER,
    stage3_passed       INTEGER,
    error_message       TEXT,
    CONSTRAINT uq_batch_logs_date UNIQUE (batch_date)
);

-- -----------------------------------------------------------
-- 3. RLS (Row Level Security) ポリシー
--    Next.js フロントから匿名アクセスを許可する場合の設定例
--    本番ではサービスロールキーを使って制限すること
-- -----------------------------------------------------------

-- candidates: 読み取りのみ公開 (フロントエンド用)
ALTER TABLE candidates ENABLE ROW LEVEL SECURITY;

CREATE POLICY "allow_read_candidates"
    ON candidates FOR SELECT
    USING (true);

-- batch_logs: 読み取りのみ公開
ALTER TABLE batch_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "allow_read_batch_logs"
    ON batch_logs FOR SELECT
    USING (true);


-- -----------------------------------------------------------
-- 4. Supabase Edge Function / API で使うビュー
--    最新バッチ日の候補銘柄のみを返す便利ビュー
-- -----------------------------------------------------------
CREATE OR REPLACE VIEW latest_candidates AS
SELECT c.*
FROM candidates c
INNER JOIN (
    SELECT MAX(batch_date) AS latest_date FROM candidates
) ld ON c.batch_date = ld.latest_date
ORDER BY c.lgbm_win_prob DESC NULLS LAST;

COMMENT ON VIEW latest_candidates IS
    '最新バッチの候補銘柄のみを返すビュー。フロントエンドからはこのビューを参照する。';
