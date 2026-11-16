// tests_harness.rs – Shared unit tests for the matching engine

#[cfg(not(after_impl))]
#[path = "before/matching.rs"]
mod matching;

#[cfg(after_impl)]
#[path = "after/matching.rs"]
mod matching;

use matching::*;

#[test]
fn test_limit_buy_match_sell() {
    let mut eng = new_engine();
    // set account limits
    eng.account_mut(1).max_margin = 1_000_000;
    eng.account_mut(2).max_margin = 1_000_000;
    // sell order
    let sell = Order::new(2, Side::Sell, OrderType::Limit, 1000, 10, 0);
    submit_order(&mut eng, sell).unwrap();
    // buy order that matches
    let buy = Order::new(1, Side::Buy, OrderType::Limit, 1000, 10, 0);
    submit_order(&mut eng, buy).unwrap();
    // after full fill, both orders removed
    assert!(eng.orders.is_empty());
    // check position updates
    assert_eq!(eng.accounts.get(&1).unwrap().position, 10);
    assert_eq!(eng.accounts.get(&2).unwrap().position, -10);
}

#[test]
fn test_market_order_fills_partial() {
    let mut eng = new_engine();
    eng.account_mut(1).max_margin = 1_000_000;
    eng.account_mut(2).max_margin = 1_000_000;
    // two sell limit orders at different prices
    let s1 = Order::new(2, Side::Sell, OrderType::Limit, 1000, 5, 0);
    let s2 = Order::new(2, Side::Sell, OrderType::Limit, 1010, 5, 0);
    submit_order(&mut eng, s1).unwrap();
    submit_order(&mut eng, s2).unwrap();
    // market buy for 8 units – should take 5 at 1000 and 3 at 1010
    let m = Order::new(1, Side::Buy, OrderType::Market, 0, 8, 0);
    submit_order(&mut eng, m).unwrap();
    // remaining sell order qty should be 2 at 1010
    let remaining = eng.orders.values().find(|o| o.side == Side::Sell).unwrap();
    assert_eq!(remaining.qty, 2);
    // buyer position should be 8
    assert_eq!(eng.accounts.get(&1).unwrap().position, 8);
}

#[test]
fn test_ioc_partial_fill() {
    let mut eng = new_engine();
    eng.account_mut(1).max_margin = 1_000_000;
    eng.account_mut(2).max_margin = 1_000_000;
    // sell order
    let s = Order::new(2, Side::Sell, OrderType::Limit, 1000, 5, 0);
    submit_order(&mut eng, s).unwrap();
    // IOC buy for 3 – should fill partially and cancel remainder
    let ioc = Order::new(1, Side::Buy, OrderType::IOC, 1000, 3, 0);
    submit_order(&mut eng, ioc).unwrap();
    // sell order should have qty 2 left
    let remaining = eng.orders.values().find(|o| o.side == Side::Sell).unwrap();
    assert_eq!(remaining.qty, 2);
    // no IOC order remains
    assert!(!eng.orders.values().any(|o| o.id == 2 && o.typ == OrderType::IOC));
}

#[test]
fn test_fok_full_fill_success() {
    let mut eng = new_engine();
    eng.account_mut(1).max_margin = 1_000_000;
    eng.account_mut(2).max_margin = 1_000_000;
    // sell 5 @1000 and 5 @1010
    submit_order(&mut eng, Order::new(2, Side::Sell, OrderType::Limit, 1000, 5, 0)).unwrap();
    submit_order(&mut eng, Order::new(2, Side::Sell, OrderType::Limit, 1010, 5, 0)).unwrap();
    // FOK buy for 10, price limit 1010 – should fully fill
    let fok = Order::new(1, Side::Buy, OrderType::FOK, 1010, 10, 0);
    submit_order(&mut eng, fok).unwrap();
    // all orders cleared
    assert!(eng.orders.is_empty());
}

#[test]
fn test_fok_not_fillable_reject() {
    let mut eng = new_engine();
    eng.account_mut(1).max_margin = 1_000_000;
    // only 5 available
    submit_order(&mut eng, Order::new(2, Side::Sell, OrderType::Limit, 1000, 5, 0)).unwrap();
    // FOK buy for 10 – should reject
    let fok = Order::new(1, Side::Buy, OrderType::FOK, 1000, 10, 0);
    let res = submit_order(&mut eng, fok);
    assert!(res.is_err());
    // original sell remains
    assert_eq!(eng.orders.len(), 1);
}

#[test]
fn test_self_trade_prevention() {
    let mut eng = new_engine();
    eng.account_mut(1).max_margin = 1_000_000;
    // first order (sell)
    let s = Order::new(1, Side::Sell, OrderType::Limit, 1000, 5, 0);
    submit_order(&mut eng, s).unwrap();
    // second order (buy) same account, self_trade flag true
    let b = Order::new(1, Side::Buy, OrderType::Limit, 1000, 5, 0);
    let res = submit_order(&mut eng, b);
    assert!(res.is_err()); // rejected due to self‑trade
}

#[test]
fn test_modify_order_price_and_qty() {
    let mut eng = new_engine();
    eng.account_mut(1).max_margin = 1_000_000;
    let o = Order::new(1, Side::Buy, OrderType::Limit, 1000, 5, 0);
    submit_order(&mut eng, o).unwrap();
    let oid = *eng.orders.iter().find(|(_, o)| o.side == Side::Buy).unwrap().0;
    modify_order(&mut eng, oid, Some(1010), Some(10)).unwrap();
    let modified = eng.orders.get(&oid).unwrap();
    assert_eq!(modified.price, 1010);
    assert_eq!(modified.qty, 10);
    // ensure order sits at new price level
    assert!(eng.bids.contains_key(&1010));
}

#[test]
fn test_cancel_order() {
    let mut eng = new_engine();
    eng.account_mut(1).max_margin = 1_000_000;
    let o = Order::new(1, Side::Buy, OrderType::Limit, 1000, 5, 0);
    submit_order(&mut eng, o).unwrap();
    let oid = *eng.orders.iter().find(|(_, o)| o.side == Side::Buy).unwrap().0;
    cancel_order(&mut eng, oid).unwrap();
    assert!(!eng.orders.contains_key(&oid));
    // order book should be empty
    assert!(eng.bids.is_empty());
}
