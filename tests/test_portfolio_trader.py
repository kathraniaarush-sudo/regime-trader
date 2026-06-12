"""Order-reconciliation logic for the live portfolio trader (no broker calls)."""
import pandas as pd

from regime_trader.portfolio_trader import PortfolioTrader


PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 25.0})


def test_enters_target_from_cash():
    # 50% AAA of $100k at $100 -> 500 shares; 50% BBB at $50 -> 1000 shares
    orders = PortfolioTrader.plan_orders(
        {"AAA": 0.5, "BBB": 0.5}, equity=100_000, prices=PRICES, current_shares={})
    by_sym = {o[0]: o for o in orders}
    assert by_sym["AAA"] == ("AAA", "buy", 500)
    assert by_sym["BBB"] == ("BBB", "buy", 1000)


def test_exits_names_not_in_target():
    # holding CCC but target doesn't include it -> sell all
    orders = PortfolioTrader.plan_orders(
        {"AAA": 1.0}, equity=100_000, prices=PRICES, current_shares={"CCC": 400})
    assert ("CCC", "sell", 400) in orders


def test_sells_ordered_before_buys():
    orders = PortfolioTrader.plan_orders(
        {"AAA": 1.0}, equity=100_000, prices=PRICES, current_shares={"BBB": 1000})
    sides = [o[1] for o in orders]
    assert sides == sorted(sides, key=lambda s: 0 if s == "sell" else 1)
    assert sides[0] == "sell"


def test_skips_dust_trades():
    # already almost aligned: target 500 AAA, holding 499 -> 1 share = $100 > min,
    # but holding 500 exactly -> no order
    aligned = PortfolioTrader.plan_orders(
        {"AAA": 0.5}, equity=100_000, prices=PRICES, current_shares={"AAA": 500})
    assert aligned == []


def test_cash_target_closes_everything():
    orders = PortfolioTrader.plan_orders(
        {}, equity=100_000, prices=PRICES, current_shares={"AAA": 100, "BBB": 200})
    assert {o[0] for o in orders} == {"AAA", "BBB"}
    assert all(o[1] == "sell" for o in orders)
