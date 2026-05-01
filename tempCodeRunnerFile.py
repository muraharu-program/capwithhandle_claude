log.info(f"Inserting {len(rows)} candidates into Supabase...")
# 100件ずつバッチインサート
for i in range(0, len(rows), 100):
    supabase.table("candidates").insert(rows[i:i+100]).execute()
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

    # バッチ開始ログを記録
    supabase.table("batch_logs").upsert({
        "batch_date": str(batch_date),
        "started_at": datetime.utcnow().isoformat(),
        "status":     "running",
    }).execute()

    try:
        # ── データ取得 ──
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

        # バッチ完了ログ
        supabase.table("batch_logs").upsert({
            "batch_date":    str(batch_date),
            "finished_at":   datetime.utcnow().isoformat(),
            "status":        "success",
            "total_tickers": len(tickers),
            "stage1_passed": len(s1_passed),
            "stage2_passed": len(s2_passed),
            "stage3_passed": len(s3_passed),
        }).execute()

        log.info(f"=== Batch complete. Final candidates: {len(s3_passed)} ===")

    except Exception as e:
        log.exception(f"Batch failed: {e}")
        supabase.table("batch_logs").upsert({
            "batch_date":    str(batch_date),
            "finished_at":   datetime.utcnow().isoformat(),
            "status":        "failed",
            "error_message": str(e),
        }).execute()
        raise


if __name__ == "__main__":
    main()
