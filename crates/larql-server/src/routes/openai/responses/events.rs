//! Typed SSE events for `/v1/responses` streaming.
//!
//! The Responses stream contract differs from chat's chunk stream: each
//! SSE frame carries an `event:` name and a JSON payload with `type`
//! and a monotonically increasing `sequence_number`. Lifecycle:
//!
//! ```text
//! response.created → response.in_progress
//!   → response.output_item.added → response.content_part.added
//!     → response.output_text.delta (per token)
//!   → response.output_text.done → response.content_part.done
//!   → response.output_item.done
//! → response.completed            (or response.failed)
//! data: [DONE]
//! ```

use super::types::{OutputContent, OutputItem, ResponseObject};

pub(super) const EV_CREATED: &str = "response.created";
pub(super) const EV_IN_PROGRESS: &str = "response.in_progress";
pub(super) const EV_COMPLETED: &str = "response.completed";
pub(super) const EV_FAILED: &str = "response.failed";
pub(super) const EV_OUTPUT_ITEM_ADDED: &str = "response.output_item.added";
pub(super) const EV_OUTPUT_ITEM_DONE: &str = "response.output_item.done";
pub(super) const EV_CONTENT_PART_ADDED: &str = "response.content_part.added";
pub(super) const EV_CONTENT_PART_DONE: &str = "response.content_part.done";
pub(super) const EV_OUTPUT_TEXT_DELTA: &str = "response.output_text.delta";
pub(super) const EV_OUTPUT_TEXT_DONE: &str = "response.output_text.done";

/// One wire frame: SSE `event:` name + `data:` JSON payload.
pub(super) type EventFrame = (&'static str, String);

/// Builds the typed event frames, threading `sequence_number` through
/// every payload so client SDKs can detect gaps/reordering.
pub(super) struct EventSeq {
    next_seq: u64,
}

impl EventSeq {
    pub(super) fn new() -> Self {
        Self { next_seq: 0 }
    }

    fn frame(
        &mut self,
        kind: &'static str,
        mut payload: serde_json::Map<String, serde_json::Value>,
    ) -> EventFrame {
        payload.insert("type".into(), kind.into());
        payload.insert("sequence_number".into(), self.next_seq.into());
        self.next_seq += 1;
        (kind, serde_json::Value::Object(payload).to_string())
    }

    fn response_frame(&mut self, kind: &'static str, response: &ResponseObject) -> EventFrame {
        let mut payload = serde_json::Map::new();
        payload.insert(
            "response".into(),
            serde_json::to_value(response).unwrap_or(serde_json::Value::Null),
        );
        self.frame(kind, payload)
    }

    pub(super) fn created(&mut self, response: &ResponseObject) -> EventFrame {
        self.response_frame(EV_CREATED, response)
    }

    pub(super) fn in_progress(&mut self, response: &ResponseObject) -> EventFrame {
        self.response_frame(EV_IN_PROGRESS, response)
    }

    pub(super) fn completed(&mut self, response: &ResponseObject) -> EventFrame {
        self.response_frame(EV_COMPLETED, response)
    }

    pub(super) fn failed(&mut self, response: &ResponseObject) -> EventFrame {
        self.response_frame(EV_FAILED, response)
    }

    pub(super) fn output_item_added(
        &mut self,
        output_index: usize,
        item: &OutputItem,
    ) -> EventFrame {
        let mut payload = serde_json::Map::new();
        payload.insert("output_index".into(), output_index.into());
        payload.insert(
            "item".into(),
            serde_json::to_value(item).unwrap_or(serde_json::Value::Null),
        );
        self.frame(EV_OUTPUT_ITEM_ADDED, payload)
    }

    pub(super) fn output_item_done(
        &mut self,
        output_index: usize,
        item: &OutputItem,
    ) -> EventFrame {
        let mut payload = serde_json::Map::new();
        payload.insert("output_index".into(), output_index.into());
        payload.insert(
            "item".into(),
            serde_json::to_value(item).unwrap_or(serde_json::Value::Null),
        );
        self.frame(EV_OUTPUT_ITEM_DONE, payload)
    }

    pub(super) fn content_part_added(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        part: &OutputContent,
    ) -> EventFrame {
        let mut payload = content_locator(item_id, output_index, content_index);
        payload.insert(
            "part".into(),
            serde_json::to_value(part).unwrap_or(serde_json::Value::Null),
        );
        self.frame(EV_CONTENT_PART_ADDED, payload)
    }

    pub(super) fn content_part_done(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        part: &OutputContent,
    ) -> EventFrame {
        let mut payload = content_locator(item_id, output_index, content_index);
        payload.insert(
            "part".into(),
            serde_json::to_value(part).unwrap_or(serde_json::Value::Null),
        );
        self.frame(EV_CONTENT_PART_DONE, payload)
    }

    pub(super) fn output_text_delta(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        delta: &str,
    ) -> EventFrame {
        let mut payload = content_locator(item_id, output_index, content_index);
        payload.insert("delta".into(), delta.into());
        self.frame(EV_OUTPUT_TEXT_DELTA, payload)
    }

    pub(super) fn output_text_done(
        &mut self,
        item_id: &str,
        output_index: usize,
        content_index: usize,
        text: &str,
    ) -> EventFrame {
        let mut payload = content_locator(item_id, output_index, content_index);
        payload.insert("text".into(), text.into());
        self.frame(EV_OUTPUT_TEXT_DONE, payload)
    }
}

fn content_locator(
    item_id: &str,
    output_index: usize,
    content_index: usize,
) -> serde_json::Map<String, serde_json::Value> {
    let mut payload = serde_json::Map::new();
    payload.insert("item_id".into(), item_id.into());
    payload.insert("output_index".into(), output_index.into());
    payload.insert("content_index".into(), content_index.into());
    payload
}
