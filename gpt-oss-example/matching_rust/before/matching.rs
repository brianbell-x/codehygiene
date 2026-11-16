// matching.rs – Stand‑alone order‑book matching & pre‑trade risk engine
// No external crates, only the Rust standard library. Designed to compile with `rustc` and pass the supplied unit tests.

use std::collections::{BTreeMap, VecDeque};

pub type OrderId = u64;
pub type AccountId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side { Buy, Sell }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType { Limit, Market, IOC, FOK }

#[derive(Debug, Clone)]
pub struct Order {
    pub id: OrderId,
    pub account: AccountId,
    pub side: Side,
    pub typ: OrderType,
    pub price: u64,          // price in ticks (0 for market orders)
    pub qty: u64,            // remaining quantity
    pub orig_qty: u64,
    pub fee_tier: u8,
    pub is_self_trade: bool,
    pub time: u64,
}

#[derive(Debug, Clone)]
pub struct Fill {
    pub aggressor_id: OrderId,
    pub maker_id: OrderId,
    pub price: u64,
    pub qty: u64,
    pub fee_paid: i64, // positive = fee, negative = rebate
}

#[derive(Debug, Clone)]
pub enum Event {
    NewOrder(OrderId),
    Cancel(OrderId),
    Modify(OrderId),
    Fill(Fill),
    Reject(OrderId, &'static str),
}

#[derive(Debug, Default, Clone)]
pub struct AccountRisk {
    pub margin_used: i64,
    pub position: i64,
    pub max_margin: i64,
    pub max_position: i64,
}

pub const FEE_TIERS: [i64; 4] = [10, 5, -5, -10]; // bps, positive = fee, negative = rebate

type Queue = VecDeque<OrderId>;
type BookSide = BTreeMap<u64, Queue>; // price -> FIFO queue of order ids

pub struct Engine {
    pub next_order_id: OrderId,
    pub next_timestamp: u64,
    pub orders: BTreeMap<OrderId, Order>,
    pub accounts: BTreeMap<AccountId, AccountRisk>,
    pub bids: BookSide, // highest price first (iterate rev())
    pub asks: BookSide, // lowest price first
    pub events: Vec<Event>,
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            next_order_id: 1,
            next_timestamp: 1,
            orders: BTreeMap::new(),
            accounts: BTreeMap::new(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            events: Vec::new(),
        }
    }

    // expose mutable account risk, inserting defaults when missing
    pub fn account_mut(&mut self, acct: AccountId) -> &mut AccountRisk {
        self.accounts.entry(acct).or_default()
    }

    pub fn submit(&mut self, mut order: Order) -> Result<(), &'static str> {
        // assign identifiers
        order.id = self.next_order_id;
        self.next_order_id += 1;
        order.time = self.next_timestamp;
        self.next_timestamp += 1;
        order.orig_qty = order.qty;

        // simple margin check for buys (sell side shorting ignored for brevity)
        if order.side == Side::Buy {
            let required = (order.price as i64) * (order.qty as i64);
            let acct = self.account_mut(order.account);
            if acct.margin_used + required > acct.max_margin {
                self.events.push(Event::Reject(order.id, "margin limit"));
                return Err("margin limit");
            }
            acct.margin_used += required;
        }

        // Dispatch based on order type. For Limit we attempt matching first.
        let result = match order.typ {
            OrderType::Limit => self.match_order(order),
            OrderType::Market => self.match_order(order),
            OrderType::IOC => self.match_order(order),
            OrderType::FOK => {
                if self.can_fully_fill(&order) {
                    self.match_order(order)
                } else {
                    self.events.push(Event::Reject(order.id, "FOK not fillable"));
                    Err("FOK not fillable")
                }
            }
        };
        result
    }

    fn add_to_book(&mut self, order: Order) {
        let id = order.id;
        self.events.push(Event::NewOrder(id));
        self.orders.insert(id, order.clone());
        let side_map = match order.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        side_map.entry(order.price).or_insert_with(VecDeque::new).push_back(id);
    }

    fn match_order(&mut self, mut incoming: Order) -> Result<(), &'static str> {
        let price_levels: Vec<u64> = match incoming.side {
            Side::Buy => self.asks.keys().cloned().collect(), // lowest ask first
            Side::Sell => self.bids.keys().rev().cloned().collect(), // highest bid first
        };

        for price in price_levels {
            if incoming.qty == 0 { break; }
            // price‑time priority check for limit/IOC/FOK
            let price_ok = match incoming.typ {
                OrderType::Market => true,
                _ => match incoming.side {
                    Side::Buy => incoming.price >= price,
                    Side::Sell => incoming.price <= price,
                },
            };
            if !price_ok { break; }
            // clone maker ids to avoid mutable borrow conflicts
            let maker_ids: Vec<OrderId> = match incoming.side {
                Side::Buy => self.asks.get(&price).map(|q| q.iter().cloned().collect()).unwrap_or_default(),
                Side::Sell => self.bids.get(&price).map(|q| q.iter().cloned().collect()).unwrap_or_default(),
            };

            for maker_id in maker_ids {
                if incoming.qty == 0 { break; }
                let maker = match self.orders.get(&maker_id) { Some(m) => m.clone(), None => continue };
                // self‑trade protection
                if maker.account == incoming.account && incoming.is_self_trade {
                    self.events.push(Event::Reject(incoming.id, "self‑trade prevented"));
                    // undo margin reservation for the incoming buy order if any
                    if incoming.side == Side::Buy {
                        let acct = self.account_mut(incoming.account);
                        let reserved = (incoming.price as i64) * (incoming.qty as i64);
                        acct.margin_used -= reserved;
                    }
                    return Err("self‑trade prevented");
                }
                let trade_qty = incoming.qty.min(maker.qty);
                let fee = (FEE_TIERS[incoming.fee_tier as usize] * trade_qty as i64) / 10_000;
                let fill = Fill { aggressor_id: incoming.id, maker_id, price, qty: trade_qty, fee_paid: fee };
                self.events.push(Event::Fill(fill.clone()));
                // update risk for both sides
                self.update_risk(incoming.account, incoming.side, price, trade_qty, fee);
                self.update_risk(maker.account, maker.side, price, trade_qty, -fee);
                // adjust quantities
                incoming.qty -= trade_qty;
                // update maker order
                if let Some(maker_mut) = self.orders.get_mut(&maker_id) {
                    maker_mut.qty -= trade_qty;
                    if maker_mut.qty == 0 {
                        self.remove_order_from_book(maker.side, price, maker_id);
                        self.events.push(Event::Cancel(maker_id));
                        self.orders.remove(&maker_id);
                    }
                }
                if incoming.typ == OrderType::IOC { break; }
            }
            // clean empty price level
            match incoming.side {
                Side::Buy => { if let Some(q) = self.asks.get(&price) { if q.is_empty() { self.asks.remove(&price); } } }
                Side::Sell => { if let Some(q) = self.bids.get(&price) { if q.is_empty() { self.bids.remove(&price); } } }
            }
        }

        // leftover handling – for Limit orders the remainder should be added back to the book
        if incoming.qty > 0 {
            match incoming.typ {
                OrderType::Market => { self.events.push(Event::Reject(incoming.id, "market unfilled")); }
                OrderType::IOC => { self.events.push(Event::Cancel(incoming.id)); }
                OrderType::Limit => { self.add_to_book(incoming); }
                _ => {}
            }
        }
        Ok(())
    }

    fn remove_order_from_book(&mut self, side: Side, price: u64, order_id: OrderId) {
        let side_map = match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        if let Some(queue) = side_map.get_mut(&price) {
            queue.retain(|&id| id != order_id);
            if queue.is_empty() { side_map.remove(&price); }
        }
    }

    fn update_risk(&mut self, acct: AccountId, side: Side, price: u64, qty: u64, fee: i64) {
        let a = self.account_mut(acct);
        let signed = match side { Side::Buy => qty as i64, Side::Sell => -(qty as i64) };
        a.position += signed;
        a.margin_used -= (price as i64) * (qty as i64);
        a.margin_used += fee; // treat fee as additional margin usage (simplified)
    }

    fn can_fully_fill(&self, order: &Order) -> bool {
        let mut needed = order.qty;
        // Choose the opposite side book
        let book = match order.side {
            Side::Buy => &self.asks,
            Side::Sell => &self.bids,
        };
        // Iterate price levels respecting price limit
        let price_iter: Vec<u64> = match order.side {
            Side::Buy => book.keys().cloned().collect(), // lowest asks first
            Side::Sell => book.keys().rev().cloned().collect(), // highest bids first
        };
        for price in price_iter {
            // price check same as match_order
            let price_ok = match order.typ {
                OrderType::Market => true,
                _ => match order.side {
                    Side::Buy => order.price >= price,
                    Side::Sell => order.price <= price,
                },
            };
            if !price_ok { break; }
            if let Some(queue) = book.get(&price) {
                for maker_id in queue {
                    if let Some(maker) = self.orders.get(maker_id) {
                        let take = needed.min(maker.qty);
                        needed -= take;
                        if needed == 0 { return true; }
                    }
                }
            }
        }
        false
    }

    pub fn cancel(&mut self, order_id: OrderId) -> Result<(), &'static str> {
        let order = self.orders.remove(&order_id).ok_or("order not found")?;
        // remove from book side
        let side_map = match order.side { Side::Buy => &mut self.bids, Side::Sell => &mut self.asks };
        if let Some(queue) = side_map.get_mut(&order.price) {
            queue.retain(|&id| id != order_id);
            if queue.is_empty() { side_map.remove(&order.price); }
        }
        // refund margin / adjust position
        let acct = self.account_mut(order.account);
        if order.side == Side::Buy {
            acct.margin_used -= (order.price as i64) * (order.qty as i64);
        } else {
            acct.position += order.qty as i64; // undo short position assumption
        }
        self.events.push(Event::Cancel(order_id));
        Ok(())
    }

    pub fn modify(&mut self, order_id: OrderId, new_price: Option<u64>, new_qty: Option<u64>) -> Result<(), &'static str> {
        let order = self.orders.get_mut(&order_id).ok_or("order not found")?;
        // remove from old price level
        let side_map = match order.side { Side::Buy => &mut self.bids, Side::Sell => &mut self.asks };
        if let Some(queue) = side_map.get_mut(&order.price) {
            queue.retain(|&id| id != order_id);
            if queue.is_empty() { side_map.remove(&order.price); }
        }
        // apply changes
        if let Some(p) = new_price { order.price = p; }
        if let Some(q) = new_qty { order.qty = q; order.orig_qty = q; }
        // re‑insert at new price
        side_map.entry(order.price).or_insert_with(VecDeque::new).push_back(order_id);
        self.events.push(Event::Modify(order_id));
        Ok(())
    }
}

impl Order {
    pub fn new(account: AccountId, side: Side, typ: OrderType, price: u64, qty: u64, fee_tier: u8) -> Self {
        Order { id: 0, account, side, typ, price, qty, orig_qty: qty, fee_tier, is_self_trade: true, time: 0 }
    }
}

// Public helper API used by the test harness
pub fn new_engine() -> Engine { Engine::new() }
pub fn submit_order(engine: &mut Engine, order: Order) -> Result<(), &'static str> { engine.submit(order) }
pub fn cancel_order(engine: &mut Engine, id: OrderId) -> Result<(), &'static str> { engine.cancel(id) }
pub fn modify_order(engine: &mut Engine, id: OrderId, price: Option<u64>, qty: Option<u64>) -> Result<(), &'static str> { engine.modify(id, price, qty) }
