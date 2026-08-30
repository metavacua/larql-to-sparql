//! SSE serving path for `/v1/responses` — typed events over the shared
//! generation engine.
//!
//! Wire shape: each frame is `event: <type>\ndata: <payload JSON>\n\n`
//! with a payload-embedded `sequence_number`; the stream ends with
//! `data: [DONE]`. Text deltas flow token-by-token; a function-call
//! output is emitted as one added/done item pair once the constrained
//! output has parsed (the same buffered-emission contract as the chat
//! endpoint's tool_calls chunk).

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

use crate::state::AppState;

use super::super::util::{SSE_CHANNEL_DEPTH, SSE_DONE};
use super::engine;
use super::events::{EventFrame, EventSeq};
use super::handler::{build_envelope, build_output, persist, usage_of, RequestPlan};
use super::types::{OutputContent, OutputItem, STATUS_FAILED, STATUS_IN_PROGRESS};

/// Build the SSE response; generation runs on a blocking thread and
/// feeds typed frames through a bounded channel.
pub(super) fn sse_response(
    state: Arc<AppState>,
    plan: RequestPlan,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<EventFrame>(SSE_CHANNEL_DEPTH);

    tokio::task::spawn_blocking(move || stream_worker(&state, &plan, &tx));

    let stream = ReceiverStream::new(rx)
        .map(|(name, data)| Event::default().event(name).data(data))
        .chain(tokio_stream::once(Event::default().data(SSE_DONE)))
        .map(Ok::<_, Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// The blocking worker: lifecycle events, generation, terminal event,
/// and (on success) persistence for `previous_response_id` chaining.
fn stream_worker(state: &AppState, plan: &RequestPlan, tx: &tokio::sync::mpsc::Sender<EventFrame>) {
    let started = std::time::Instant::now();
    let _gen_guard = Arc::clone(&state.runtime).enter_generation();
    let mut seq = EventSeq::new();

    let opening = build_envelope(plan, STATUS_IN_PROGRESS, None, Vec::new(), None);
    if tx.blocking_send(seq.created(&opening)).is_err() {
        return;
    }
    if tx.blocking_send(seq.in_progress(&opening)).is_err() {
        return;
    }

    // For plain text output the message item and its text part open
    // before the first delta. Function-call output opens nothing yet —
    // the item only exists once the full JSON has parsed.
    let message_id = format!(
        "{}{}",
        super::types::MESSAGE_ID_PREFIX,
        super::super::util::new_id_suffix()
    );
    if !plan.tools_active {
        let opening_item = OutputItem::Message {
            id: message_id.clone(),
            status: STATUS_IN_PROGRESS.to_string(),
            role: super::super::prompt::ASSISTANT_ROLE,
            content: Vec::new(),
        };
        let empty_part = OutputContent::OutputText {
            text: String::new(),
            annotations: Vec::new(),
        };
        if tx
            .blocking_send(seq.output_item_added(0, &opening_item))
            .is_err()
        {
            return;
        }
        if tx
            .blocking_send(seq.content_part_added(&message_id, 0, 0, &empty_part))
            .is_err()
        {
            return;
        }
    }

    let tools_active = plan.tools_active;
    let mut client_gone = false;
    let resume = super::handler::take_kv_resume(state, plan);
    let outcome = {
        let mut on_token = |text: &str| -> bool {
            if tools_active {
                // Deltas for tool output are withheld; the parsed call
                // is emitted whole after generation.
                return true;
            }
            let frame = seq.output_text_delta(&message_id, 0, 0, text);
            if tx.blocking_send(frame).is_err() {
                client_gone = true;
                return false;
            }
            true
        };
        engine::generate(
            &plan.engine,
            &plan.messages,
            plan.max_tokens,
            plan.sampling,
            &plan.stop_strings,
            plan.schema.clone(),
            resume,
            &mut on_token,
        )
    };
    if client_gone {
        return;
    }

    let mut outcome = match outcome {
        Ok(o) => o,
        Err(e) => {
            send_failed(plan, &mut seq, tx, &e.to_string());
            return;
        }
    };
    state
        .runtime
        .record(outcome.tally.into_sample(crate::state::elapsed_ms(started)));
    super::handler::retain_kv_handoff(state, plan, &mut outcome);
    let (mut output, status, incomplete) = match build_output(plan.tools_active, &outcome) {
        Ok(parts) => parts,
        Err(e) => {
            send_failed(plan, &mut seq, tx, &e);
            return;
        }
    };
    // Re-key the message item to the id the deltas streamed under so
    // added/delta/done correlate for clients tracking item lifecycles.
    if let Some(OutputItem::Message { id, .. }) = output.first_mut() {
        *id = message_id.clone();
    }

    if plan.tools_active {
        // One added/done pair carrying the whole function call.
        for item in &output {
            let _ = tx.blocking_send(seq.output_item_added(0, item));
            let _ = tx.blocking_send(seq.output_item_done(0, item));
        }
    } else {
        let final_part = OutputContent::OutputText {
            text: outcome.text.clone(),
            annotations: Vec::new(),
        };
        let _ = tx.blocking_send(seq.output_text_done(&message_id, 0, 0, &outcome.text));
        let _ = tx.blocking_send(seq.content_part_done(&message_id, 0, 0, &final_part));
        for item in &output {
            let _ = tx.blocking_send(seq.output_item_done(0, item));
        }
    }

    let envelope = build_envelope(plan, status, incomplete, output, Some(usage_of(&outcome)));
    persist(state, plan, &envelope);
    let _ = tx.blocking_send(seq.completed(&envelope));
}

fn send_failed(
    plan: &RequestPlan,
    seq: &mut EventSeq,
    tx: &tokio::sync::mpsc::Sender<EventFrame>,
    message: &str,
) {
    let mut envelope = build_envelope(plan, STATUS_FAILED, None, Vec::new(), None);
    envelope.error = Some(serde_json::json!({
        "message": message,
        "type": "server_error",
    }));
    let _ = tx.blocking_send(seq.failed(&envelope));
}
