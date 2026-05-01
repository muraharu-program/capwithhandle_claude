// =============================================================
// Next.js + TypeScript: カップウィズハンドル スクリーナー UI
// Supabase からデータを取得し TradingView Lightweight Charts で描画
// =============================================================

// ─────────────────────────────────────────────
// 1. 型定義 (Python の Supabase Insert との契約)
// ─────────────────────────────────────────────

/** TradingView Lightweight Charts に渡す OHLCV の 1本分 */
export interface OhlcvBar {
  time: string;   // "YYYY-MM-DD"
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/** LightGBM に渡した特徴量スナップショット */
export interface FeaturesSnapshot {
  vol_spike_ratio: number;
  volatility_contraction: number;
  rs_rank: number;
  momentum_20d: number;
  cup_symmetry_score: number;
  cup_depth_pct: number;
  handle_depth_pct: number;
  breakout_vol_ratio: number;
  price_vs_ma200_pct: number;
  prior_uptrend_pct: number;
  cup_duration_weeks: number;
  handle_duration_days: number;
}

/** candidates テーブルの 1行 = 1候補銘柄 */
export interface Candidate {
  id: number;
  ticker: string;
  company_name: string | null;
  screened_at: string;            // ISO 8601
  batch_date: string;             // "YYYY-MM-DD"

  // 現在値
  current_price: number;
  volume_today: number | null;
  market_cap: number | null;
  sector: string | null;
  industry: string | null;

  // カップ形状
  cup_start_date: string | null;
  cup_end_date: string | null;
  cup_left_high: number | null;
  cup_right_high: number | null;
  cup_bottom: number | null;
  cup_depth_pct: number | null;
  cup_duration_weeks: number | null;
  cup_symmetry_score: number | null;

  // ハンドル形状
  handle_start_date: string | null;
  handle_end_date: string | null;
  handle_high: number | null;
  handle_low: number | null;
  handle_depth_pct: number | null;
  handle_duration_days: number | null;
  handle_volume_decline_pct: number | null;

  // ブレイクアウト
  breakout_price: number | null;
  is_breaking_out: boolean;
  breakout_volume_ratio: number | null;

  // トレンド
  prior_uptrend_pct: number | null;
  price_vs_ma200_pct: number | null;

  // 利確・損切り
  take_profit_a: number | null;
  take_profit_b: number | null;
  stop_loss: number | null;

  // ML スコア
  lgbm_win_prob: number | null;
  funnel_stage: number;

  // JSON フィールド
  features_json: FeaturesSnapshot | null;
  ohlcv_json: OhlcvBar[] | null;
}

// ─────────────────────────────────────────────
// 2. Supabase クライアント初期化
// ─────────────────────────────────────────────
// lib/supabase.ts

import { createClient } from "@supabase/supabase-js";

const supabaseUrl  = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnon);

// ─────────────────────────────────────────────
// 3. データフェッチ関数
// ─────────────────────────────────────────────
// lib/fetchCandidates.ts

/**
 * latest_candidates ビューから最新バッチの候補銘柄を取得。
 * @param onlyBreakout  true の場合、ブレイクアウト中の銘柄のみ返す
 * @param minWinProb    LightGBM 勝率の下限フィルター (0.0~1.0)
 */
export async function fetchCandidates(
  onlyBreakout = false,
  minWinProb = 0.0
): Promise<Candidate[]> {
  let query = supabase
    .from("latest_candidates")   // latest_candidates ビューを参照
    .select("*")
    .order("lgbm_win_prob", { ascending: false });

  if (onlyBreakout) {
    query = query.eq("is_breaking_out", true);
  }

  if (minWinProb > 0) {
    query = query.gte("lgbm_win_prob", minWinProb);
  }

  const { data, error } = await query;

  if (error) {
    console.error("Supabase fetch error:", error.message);
    throw new Error(error.message);
  }

  return (data ?? []) as Candidate[];
}

/**
 * 特定のティッカーの詳細を 1件取得。
 */
export async function fetchCandidateByTicker(
  ticker: string
): Promise<Candidate | null> {
  const { data, error } = await supabase
    .from("latest_candidates")
    .select("*")
    .eq("ticker", ticker.toUpperCase())
    .maybeSingle();

  if (error) throw new Error(error.message);
  return data as Candidate | null;
}

// ─────────────────────────────────────────────
// 4. チャートコンポーネント (TradingView Lightweight Charts)
// ─────────────────────────────────────────────
// components/CandlestickChart.tsx

"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  LineData,
  ColorType,
  CrosshairMode,
} from "lightweight-charts";
import type { Candidate, OhlcvBar } from "@/lib/types";

interface Props {
  candidate: Candidate;
}

/**
 * カップウィズハンドルのチャートを描画するコンポーネント。
 * - ローソク足 (OHLCV)
 * - ブレイクアウトライン (水平線)
 * - 利確ライン A / B (水平線)
 * - 損切りライン (水平線)
 */
export function CandlestickChart({ candidate }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // ── チャート初期化 ──
    const chart: IChartApi = createChart(containerRef.current, {
      width:  containerRef.current.clientWidth,
      height: 420,
      layout: {
        background: { type: ColorType.Solid, color: "#0f172a" },
        textColor:  "#94a3b8",
      },
      grid: {
        vertLines:  { color: "#1e293b" },
        horzLines:  { color: "#1e293b" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#334155" },
      timeScale: {
        borderColor:    "#334155",
        timeVisible:    true,
        secondsVisible: false,
      },
    });
    chartRef.current = chart;

    // ── ローソク足シリーズ ──
    const candleSeries: ISeriesApi<"Candlestick"> =
      chart.addCandlestickSeries({
        upColor:   "#22c55e",
        downColor: "#ef4444",
        borderUpColor:   "#22c55e",
        borderDownColor: "#ef4444",
        wickUpColor:   "#22c55e",
        wickDownColor: "#ef4444",
      });

    const ohlcv = candidate.ohlcv_json ?? [];
    const candleData: CandlestickData[] = ohlcv.map((bar: OhlcvBar) => ({
      time:  bar.time as unknown as CandlestickData["time"],
      open:  bar.open,
      high:  bar.high,
      low:   bar.low,
      close: bar.close,
    }));
    candleSeries.setData(candleData);

    // ── 水平線ヘルパー ──
    const addHorizontalLine = (
      price: number,
      color: string,
      lineWidth: 1 | 2 | 3 | 4,
      title: string
    ) => {
      if (!ohlcv.length) return;
      const firstTime = ohlcv[0].time;
      const lastTime  = ohlcv[ohlcv.length - 1].time;

      const lineSeries = chart.addLineSeries({
        color,
        lineWidth,
        lastValueVisible: true,
        priceLineVisible: false,
        title,
      });
      const lineData: LineData[] = [
        { time: firstTime as unknown as LineData["time"], value: price },
        { time: lastTime  as unknown as LineData["time"], value: price },
      ];
      lineSeries.setData(lineData);
    };

    // ── ブレイクアウトライン ──
    if (candidate.breakout_price) {
      addHorizontalLine(
        candidate.breakout_price,
        "#f59e0b",   // 黄色
        2,
        "BO"
      );
    }

    // ── 利確ライン A ──
    if (candidate.take_profit_a) {
      addHorizontalLine(
        candidate.take_profit_a,
        "#34d399",   // 緑
        1,
        "TP-A"
      );
    }

    // ── 利確ライン B ──
    if (candidate.take_profit_b) {
      addHorizontalLine(
        candidate.take_profit_b,
        "#6ee7b7",   // 薄緑
        1,
        "TP-B"
      );
    }

    // ── 損切りライン ──
    if (candidate.stop_loss) {
      addHorizontalLine(
        candidate.stop_loss,
        "#f87171",   // 赤
        2,
        "SL"
      );
    }

    // レスポンシブ対応
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width });
      }
    });
    ro.observe(containerRef.current);

    chart.timeScale().fitContent();

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [candidate]);

  return <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />;
}

// ─────────────────────────────────────────────
// 5. 候補カードコンポーネント
// ─────────────────────────────────────────────
// components/CandidateCard.tsx

import type { Candidate } from "@/lib/types";

interface CardProps {
  candidate: Candidate;
  onClick?: () => void;
}

export function CandidateCard({ candidate: c, onClick }: CardProps) {
  const winPct = c.lgbm_win_prob != null
    ? `${(c.lgbm_win_prob * 100).toFixed(1)}%`
    : "N/A";

  return (
    <div
      onClick={onClick}
      className="
        cursor-pointer rounded-xl border border-slate-700 bg-slate-800
        p-4 hover:border-amber-500 transition-colors
      "
    >
      {/* ヘッダー */}
      <div className="flex items-center justify-between mb-2">
        <div>
          <span className="text-lg font-bold text-white">{c.ticker}</span>
          {c.company_name && (
            <span className="ml-2 text-xs text-slate-400">{c.company_name}</span>
          )}
        </div>
        {c.is_breaking_out && (
          <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-xs font-semibold text-amber-400">
            🚀 BREAKOUT
          </span>
        )}
      </div>

      {/* 価格グリッド */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm mt-3">
        <PriceRow label="現在値"       value={`$${c.current_price.toFixed(2)}`} />
        <PriceRow label="BO ライン"    value={c.breakout_price ? `$${c.breakout_price.toFixed(2)}` : "-"} highlight />
        <PriceRow label="利確 A"       value={c.take_profit_a  ? `$${c.take_profit_a.toFixed(2)}`  : "-"} color="text-emerald-400" />
        <PriceRow label="利確 B"       value={c.take_profit_b  ? `$${c.take_profit_b.toFixed(2)}`  : "-"} color="text-emerald-300" />
        <PriceRow label="損切り"       value={c.stop_loss      ? `$${c.stop_loss.toFixed(2)}`      : "-"} color="text-red-400" />
        <PriceRow label="勝率予測"     value={winPct} color="text-sky-400" />
      </div>

      {/* カップ・ハンドル情報 */}
      <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-400">
        <Tag label={`カップ深さ ${c.cup_depth_pct?.toFixed(1)}%`} />
        <Tag label={`カップ期間 ${c.cup_duration_weeks}週`} />
        <Tag label={`ハンドル ${c.handle_duration_days}日`} />
        {c.cup_symmetry_score != null && (
          <Tag label={`U字スコア ${(c.cup_symmetry_score * 100).toFixed(0)}`} />
        )}
      </div>
    </div>
  );
}

function PriceRow({
  label,
  value,
  color = "text-slate-200",
  highlight = false,
}: {
  label: string;
  value: string;
  color?: string;
  highlight?: boolean;
}) {
  return (
    <>
      <span className="text-slate-400">{label}</span>
      <span className={`font-mono font-medium ${highlight ? "text-amber-400" : color}`}>
        {value}
      </span>
    </>
  );
}

function Tag({ label }: { label: string }) {
  return (
    <span className="rounded bg-slate-700 px-1.5 py-0.5">{label}</span>
  );
}

// ─────────────────────────────────────────────
// 6. メインページ (App Router)
// ─────────────────────────────────────────────
// app/page.tsx

"use client";

import { useEffect, useState } from "react";
import { fetchCandidates } from "@/lib/fetchCandidates";
import { CandidateCard }   from "@/components/CandidateCard";
import { CandlestickChart } from "@/components/CandlestickChart";
import type { Candidate }  from "@/lib/types";

export default function HomePage() {
  const [candidates, setCandidates]   = useState<Candidate[]>([]);
  const [selected, setSelected]       = useState<Candidate | null>(null);
  const [loading, setLoading]         = useState(true);
  const [onlyBreakout, setOnlyBreakout] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchCandidates(onlyBreakout, 0.5)
      .then((data) => {
        setCandidates(data);
        setSelected(data[0] ?? null);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [onlyBreakout]);

  return (
    <main className="min-h-screen bg-slate-900 text-slate-100">
      {/* ナビゲーションバー */}
      <header className="border-b border-slate-700 px-6 py-4 flex items-center justify-between">
        <h1 className="text-xl font-bold text-amber-400">
          ☕ Cup with Handle Screener
        </h1>
        <div className="flex items-center gap-4 text-sm">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              className="accent-amber-500"
              checked={onlyBreakout}
              onChange={(e) => setOnlyBreakout(e.target.checked)}
            />
            ブレイクアウト中のみ
          </label>
          <span className="text-slate-400">
            {loading ? "読み込み中..." : `${candidates.length} 銘柄`}
          </span>
        </div>
      </header>

      <div className="flex h-[calc(100vh-65px)]">
        {/* 左ペイン: 候補リスト */}
        <aside className="w-80 flex-shrink-0 overflow-y-auto border-r border-slate-700 p-3 space-y-2">
          {loading ? (
            <p className="text-center text-slate-500 mt-8">読み込み中...</p>
          ) : candidates.length === 0 ? (
            <p className="text-center text-slate-500 mt-8">
              本日の候補銘柄はありません
            </p>
          ) : (
            candidates.map((c) => (
              <CandidateCard
                key={c.id}
                candidate={c}
                onClick={() => setSelected(c)}
              />
            ))
          )}
        </aside>

        {/* 右ペイン: チャート詳細 */}
        <section className="flex-1 overflow-y-auto p-6">
          {selected ? (
            <>
              <div className="mb-4 flex items-baseline gap-4">
                <h2 className="text-2xl font-bold">{selected.ticker}</h2>
                <span className="text-slate-400">{selected.company_name}</span>
                <span className="ml-auto text-slate-400 text-sm">
                  スクリーニング日: {selected.batch_date}
                </span>
              </div>

              {/* チャート */}
              <CandlestickChart candidate={selected} />

              {/* 詳細数値パネル */}
              <div className="mt-6 grid grid-cols-3 gap-4">
                <MetricPanel title="ブレイクアウト" color="amber">
                  <Metric label="BOライン"        value={`$${selected.breakout_price?.toFixed(2)}`} />
                  <Metric label="出来高比率"       value={`${selected.breakout_volume_ratio?.toFixed(2)}x`} />
                  <Metric label="BO中"            value={selected.is_breaking_out ? "✅ YES" : "⬜ NO"} />
                </MetricPanel>

                <MetricPanel title="利確・損切り" color="green">
                  <Metric label="利確 A" value={`$${selected.take_profit_a?.toFixed(2)}`} />
                  <Metric label="利確 B" value={`$${selected.take_profit_b?.toFixed(2)}`} />
                  <Metric label="損切り" value={`$${selected.stop_loss?.toFixed(2)}`}    />
                </MetricPanel>

                <MetricPanel title="AI スコア" color="sky">
                  <Metric label="勝率予測"     value={`${((selected.lgbm_win_prob ?? 0) * 100).toFixed(1)}%`} />
                  <Metric label="U字スコア"    value={`${((selected.cup_symmetry_score ?? 0) * 100).toFixed(0)}`} />
                  <Metric label="前トレンド"   value={`+${selected.prior_uptrend_pct?.toFixed(1)}%`} />
                </MetricPanel>
              </div>
            </>
          ) : (
            <p className="text-center text-slate-500 mt-20">
              左のリストから銘柄を選択してください
            </p>
          )}
        </section>
      </div>
    </main>
  );
}

// ─── サブコンポーネント ───

function MetricPanel({
  title,
  color,
  children,
}: {
  title: string;
  color: "amber" | "green" | "sky";
  children: React.ReactNode;
}) {
  const border = {
    amber: "border-amber-500/40",
    green: "border-emerald-500/40",
    sky:   "border-sky-500/40",
  }[color];

  return (
    <div className={`rounded-xl border ${border} bg-slate-800 p-4`}>
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-400">
        {title}
      </h3>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value?: string }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-slate-400">{label}</span>
      <span className="font-mono font-medium text-slate-100">{value ?? "-"}</span>
    </div>
  );
}
