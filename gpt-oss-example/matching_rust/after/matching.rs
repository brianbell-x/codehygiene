// matching.rs – Refactored matching engine (after implementation)
#![allow(dead_code)]

use std::collections::{BTreeMap, VecDeque};

pub type OrderId = u64;
pub type AccountId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Limit,
    Market,
    IOC,
    FOK,
}

#[derive(Debug, Clone)]
pub struct Order {
    pub id: OrderId,
    pub account: AccountId,
    pub side: Side,
    pub typ: OrderType,
    pub price: u64,
    pub qty: u64,
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
    pub fee_paid: i64,
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

pub const FEE_TIERS: [i64; 4] = [10, 5, -5, -10]; // bps

type Queue = VecDeque<OrderId>;
type BookSide = BTreeMap<u64, Queue>; // price → FIFO queue of order ids

pub struct Engine {
    pub next_order_id: OrderId,
    pub next_timestamp: u64,
    pub orders: BTreeMap<OrderId, Order>,
    pub accounts: BTreeMap<AccountId, AccountRisk>,
    pub bids: BookSide, // buy side, highest price first
    pub asks: BookSide, // sell side, lowest price first
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

    pub fn account_mut(&mut self, acct: AccountId) -> &mut AccountRisk {
        self.accounts.entry(acct).or_default()
    }

    pub fn submit(&mut self, mut order_obj: Order) -> Result<(), &'static str> {
        // Assign IDs and timestamps
        order_obj.id = self.next_order_id;
        self.next_order_id += 1;
        order_obj.time = self.next_timestamp;
        self.next_timestamp += 1;
        order_obj.orig_qty = order_obj.qty;

        // Margin check for buys
        if order_obj.side == Side::Buy {
            let required = (order_obj.price as i64) * (order_obj.qty as i64);
            let account_risk = self.account_mut(order_obj.account);
            if account_risk.margin_used + required > account_risk.max_margin {
                self.events.push(Event::Reject(order_obj.id, "margin limit"));
                return Err("margin limit");
            }
            account_risk.margin_used += required;
        }

        // Dispatch based on order type
        let result = match order_obj.typ {
            OrderType::Limit => self.process_limit(order_obj),
            OrderType::Market => self.process_market(order_obj),
            OrderType::IOC => self.process_ioc(order_obj),
            OrderType::FOK => self.process_fok(order_obj),
        };
        result
    }

    fn process_limit(&mut self, order_obj: Order) -> Result<(), &'static str> {
        let mut remaining = order_obj;
        self.match_against_book(&mut remaining)?;
        if remaining.qty > 0 {
            self.add_to_book(remaining);
        }
        Ok(())
    }

    fn process_market(&mut self, order_obj: Order) -> Result<(), &'static str> {
        let mut incoming = order_obj;
        self.match_against_book(&mut incoming)?;
        if incoming.qty > 0 {
            self.events.push(Event::Reject(incoming.id, "market unfilled"));
        }
        Ok(())
    }

    fn process_ioc(&mut self, order_obj: Order) -> Result<(), &'static str> {
        let mut incoming = order_obj;
        self.match_against_book(&mut incoming)?;
        if incoming.qty > 0 {
            self.events.push(Event::Cancel(incoming.id));
        }
        Ok(())
    }

    fn process_fok(&mut self, order_obj: Order) -> Result<(), &'static str> {
        if !self.can_fully_fill(&order_obj) {
            self.events.push(Event::Reject(order_obj.id, "FOK not fillable"));
            return Err("FOK not fillable");
        }
        self.process_limit(order_obj)
    }

    fn match_against_book(&mut self, incoming: &mut Order) -> Result<(), &'static str> {
        let price_levels: Vec<u64> = match incoming.side {
            Side::Buy => self.asks.keys().cloned().collect(), // lowest ask first
            Side::Sell => self.bids.keys().rev().cloned().collect(), // highest bid first
        };
        for level_price in price_levels {
            if incoming.qty == 0 {
                break;
            }
            if !self.price_allowed(incoming, level_price) {
                break;
            }
            self.match_at_price(incoming, level_price)?;
        }
        Ok(())
    }

    fn match_at_price(&mut self, incoming: &mut Order, level_price: u64) -> Result<(), &'static str> {
        let maker_ids = self.maker_ids_at_price(incoming.side, level_price);
        for maker_id in maker_ids {
            if incoming.qty == 0 {
                break;
            }
            self.match_maker(incoming, maker_id, level_price)?;
        }
        self.cleanup_price_level(incoming.side, level_price);
        Ok(())
    }

    fn match_maker(
        &mut self,
        incoming: &mut Order,
        maker_id: OrderId,
        level_price: u64,
    ) -> Result<(), &'static str> {
        let maker = match self.orders.get(&maker_id) {
            Some(m) => m.clone(),
            None => return Ok(()), // skip missing order
        };
        // Self‑trade guard
        if maker.account == incoming.account && incoming.is_self_trade {
            self.reject_self_trade(incoming)?;
            return Err("self‑trade prevented");
        }
        let trade_qty = incoming.qty.min(maker.qty);
        let fee = (FEE_TIERS[incoming.fee_tier as usize] * trade_qty as i64) / 10_000;
        let fill = Fill {
            aggressor_id: incoming.id,
            maker_id,
            price: level_price,
            qty: trade_qty,
            fee_paid: fee,
        };
        self.events.push(Event::Fill(fill.clone()));
        self.update_risk(
            incoming.account,
            incoming.side,
            level_price,
            trade_qty,
            fee,
        );
        self.update_risk(
            maker.account,
            maker.side,
            level_price,
            trade_qty,
            -fee,
        );
        incoming.qty -= trade_qty;
        self.update_maker(maker_id, level_price, trade_qty);
        if incoming.typ == OrderType::IOC {
            return Ok(());
        }
        Ok(())
    }

    fn reject_self_trade(&mut self, incoming: &Order) -> Result<(), &'static str> {
        self.events.push(Event::Reject(incoming.id, "self‑trade prevented"));
        if incoming.side == Side::Buy {
            let account_risk = self.account_mut(incoming.account);
            let reserved = (incoming.price as i64) * (incoming.qty as i64);
            account_risk.margin_used -= reserved;
        }
        Ok(())
    }

    fn price_allowed(&self, order_obj: &Order, level_price: u64) -> bool {
        match order_obj.typ {
            OrderType::Market => true,
            _ => match order_obj.side {
                Side::Buy => order_obj.price >= level_price,
                Side::Sell => order_obj.price <= level_price,
            },
        }
    }

    fn maker_ids_at_price(&self, side: Side, level_price: u64) -> Vec<OrderId> {
        match side {
            Side::Buy => self
                .asks
                .get(&level_price)
                .map(|q| q.iter().cloned().collect())
                .unwrap_or_default(),
            Side::Sell => self
                .bids
                .get(&level_price)
                .map(|q| q.iter().cloned().collect())
                .unwrap_or_default(),
        }
    }

    fn available_at_price(&self, side: Side, level_price: u64) -> u64 {
        let ids = self.maker_ids_at_price(side, level_price);
        let mut total = 0u64;
        for id in ids {
            if let Some(order) = self.orders.get(&id) {
                total += order.qty;
            }
        }
        total
    }

    fn update_maker(
        &mut self,
        maker_id: OrderId,
        level_price: u64,
        traded_qty: u64,
    ) {
        // Capture maker side before mutable borrow to avoid double borrowing self
        let maker_side = match self.orders.get(&maker_id) {
            Some(m) => m.side,
            None => return,
        };
        if let Some(maker) = self.orders.get_mut(&maker_id) {
            maker.qty -= traded_qty;
            if maker.qty == 0 {
                self.remove_order_from_book(maker_side, level_price, maker_id);
                self.events.push(Event::Cancel(maker_id));
                self.orders.remove(&maker_id);
            }
        }
    }

    fn cleanup_price_level(&mut self, side: Side, level_price: u64) {
        // Flattened guard clauses to avoid deep nesting
        let (map, price) = match side {
            Side::Buy => (&mut self.asks, level_price),
            Side::Sell => (&mut self.bids, level_price),
        };
        if let Some(queue) = map.get(&price) {
            if queue.is_empty() {
                map.remove(&price);
            }
        }
    }

    fn add_to_book(&mut self, order_obj: Order) {
        let order_id = order_obj.id;
        self.events.push(Event::NewOrder(order_id));
        self.orders.insert(order_id, order_obj.clone());
        let side_map = match order_obj.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        side_map
            .entry(order_obj.price)
            .or_insert_with(VecDeque::new)
            .push_back(order_id);
    }

    fn remove_order_from_book(
        &mut self,
        side: Side,
        level_price: u64,
        order_id: OrderId,
    ) {
        let side_map = match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        if let Some(queue) = side_map.get_mut(&level_price) {
            queue.retain(|&id| id != order_id);
            if queue.is_empty() {
                side_map.remove(&level_price);
            }
        }
    }

    fn update_risk(
        &mut self,
        acct: AccountId,
        side: Side,
        level_price: u64,
        qty: u64,
        fee: i64,
    ) {
        let account_risk = self.account_mut(acct);
        let signed_qty = match side {
            Side::Buy => qty as i64,
            Side::Sell => -(qty as i64),
        };
        account_risk.position += signed_qty;
        account_risk.margin_used -= (level_price as i64) * (qty as i64);
        account_risk.margin_used += fee;
    }

    fn can_fully_fill(&self, order_obj: &Order) -> bool {
        let mut needed = order_obj.qty;
        let book = match order_obj.side {
            Side::Buy => &self.asks,
            Side::Sell => &self.bids,
        };
        let price_iter: Vec<u64> = match order_obj.side {
            Side::Buy => book.keys().cloned().collect(),
            Side::Sell => book.keys().rev().cloned().collect(),
        };
        for level_price in price_iter {
            if !self.price_allowed(order_obj, level_price) {
                break;
            }
            let available = self.available_at_price(order_obj.side, level_price);
            let take = needed.min(available);
            needed -= take;
            if needed == 0 {
                return true;
            }
        }
        false
    }

    pub fn cancel(&mut self, order_id: OrderId) -> Result<(), &'static str> {
        let order_obj = self.orders.remove(&order_id).ok_or("order not found")?;
        let side_map = match order_obj.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        if let Some(queue) = side_map.get_mut(&order_obj.price) {
            queue.retain(|&id| id != order_id);
            if queue.is_empty() {
                side_map.remove(&order_obj.price);
            }
        }
        let account_risk = self.account_mut(order_obj.account);
        if order_obj.side == Side::Buy {
            account_risk.margin_used -= (order_obj.price as i64) * (order_obj.qty as i64);
        } else {
            account_risk.position += order_obj.qty as i64;
        }
        self.events.push(Event::Cancel(order_id));
        Ok(())
    }

    pub fn modify(
        &mut self,
        order_id: OrderId,
        new_price: Option<u64>,
        new_qty: Option<u64>,
    ) -> Result<(), &'static str> {
        let order_obj = self.orders.get_mut(&order_id).ok_or("order not found")?;
        let side_map = match order_obj.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        if let Some(queue) = side_map.get_mut(&order_obj.price) {
            queue.retain(|&id| id != order_id);
            if queue.is_empty() {
                side_map.remove(&order_obj.price);
            }
        }
        if let Some(p) = new_price {
            order_obj.price = p;
        }
        if let Some(q) = new_qty {
            order_obj.qty = q;
            order_obj.orig_qty = q;
        }
        side_map
            .entry(order_obj.price)
            .or_insert_with(VecDeque::new)
            .push_back(order_id);
        self.events.push(Event::Modify(order_id));
        Ok(())
    }
}

impl Order {
    pub fn new(
        account: AccountId,
        side: Side,
        typ: OrderType,
        price: u64,
        qty: u64,
        fee_tier: u8,
    ) -> Self {
        Order {
            id: 0,
            account,
            side,
            typ,
            price,
            qty,
            orig_qty: qty,
            fee_tier,
            is_self_trade: true,
            time: 0,
        }
    }
}

// Public API for tests
pub fn new_engine() -> Engine {
    Engine::new()
}

pub fn submit_order(engine: &mut Engine, order: Order) -> Result<(), &'static str> {
    engine.submit(order)
}

pub fn cancel_order(engine: &mut Engine, id: OrderId) -> Result<(), &'static str> {
    engine.cancel(id)
}

pub fn modify_order(
    engine: &mut Engine,
    id: OrderId,
    price: Option<u64>,
    qty: Option<u64>,
) -> Result<(), &'static str> {
    engine.modify(id, price, qty)
}
