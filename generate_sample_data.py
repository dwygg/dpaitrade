from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "spread",
                "atr",
                "volatility_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def gen_d1() -> list[dict]:
    rows = []
    start = datetime(2026, 1, 1, 0, 0, 0)
    price = 100.0

    # 20 根 D1，整体向上，便于识别为趋势
    for i in range(20):
        open_price = price + 0.10
        close_price = price + 0.65
        high_price = close_price + 0.35
        low_price = open_price - 0.35

        rows.append(
            {
                "timestamp": (start + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S"),
                "open": round(open_price, 5),
                "high": round(high_price, 5),
                "low": round(low_price, 5),
                "close": round(close_price, 5),
                "spread": 1.0,
                "atr": 1.2,
                "volatility_score": 0.45,
            }
        )
        price += 0.8

    return rows


def gen_h4() -> list[dict]:
    rows = []
    start = datetime(2026, 1, 17, 0, 0, 0)

    # 前 16 根上涨，后 8 根回调，便于识别为“趋势中的回调”
    for i in range(24):
        if i < 16:
            base = 100 + i * 1.2
        else:
            base = 119 - (i - 16) * 0.8

        open_price = base
        close_price = base + 0.2
        high_price = base + 0.8
        low_price = base - 0.8

        rows.append(
            {
                "timestamp": (start + timedelta(hours=4 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                "open": round(open_price, 5),
                "high": round(high_price, 5),
                "low": round(low_price, 5),
                "close": round(close_price, 5),
                "spread": 1.1,
                "atr": 1.1,
                "volatility_score": 0.42,
            }
        )

    return rows


def gen_m15() -> list[dict]:
    rows = []
    start = datetime(2026, 1, 20, 21, 0, 0)

    prev_close = 111.0

    # 生成 220 根 M15
    # 每 24 根形成一个小周期：
    # 1) 前半段缓慢抬升
    # 2) 中间小回撤
    # 3) 最后一根强突破，确保 close 能站上近端参考高点
    for i in range(220):
        block = i // 24
        phase = i % 24
        base = 111.0 + block * 0.8

        if phase < 16:
            # 缓慢上行
            close_price = base + phase * 0.045
        elif phase < 23:
            # 小回撤
            close_price = base + 0.72 - (phase - 16) * 0.035
        else:
            # 最后一根做明显突破
            # 这里故意把 close 拉高，确保能够突破前 5 根 high
            close_price = base + 1.10

        open_price = prev_close

        # 普通 K 线给小一点影线，避免 reference_high 太夸张
        if phase == 23:
            high_price = close_price + 0.05
            low_price = min(open_price, close_price) - 0.08
        else:
            high_price = max(open_price, close_price) + 0.08
            low_price = min(open_price, close_price) - 0.08

        rows.append(
            {
                "timestamp": (start + timedelta(minutes=15 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                "open": round(open_price, 5),
                "high": round(high_price, 5),
                "low": round(low_price, 5),
                "close": round(close_price, 5),
                "spread": 1.1 if phase != 5 else 1.4,
                "atr": 1.2,
                "volatility_score": 0.35 + (phase % 5) * 0.03,
            }
        )
        prev_close = close_price

    return rows

def main() -> None:
    data_dir = Path("data")
    d1_path = data_dir / "d1.csv"
    h4_path = data_dir / "h4.csv"
    m15_path = data_dir / "m15.csv"

    write_csv(d1_path, gen_d1())
    write_csv(h4_path, gen_h4())
    write_csv(m15_path, gen_m15())

    print("[信息] 示例 CSV 已生成：")
    print(f" - {d1_path}")
    print(f" - {h4_path}")
    print(f" - {m15_path}")


if __name__ == "__main__":
    main()