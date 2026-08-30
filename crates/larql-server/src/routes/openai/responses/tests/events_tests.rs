use crate::routes::openai::responses::events::*;
use crate::routes::openai::responses::types::{
    OutputContent, OutputItem, ResponseObject, RESPONSE_OBJECT, STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
};

fn envelope() -> ResponseObject {
    ResponseObject {
        id: "resp_1".into(),
        object: RESPONSE_OBJECT,
        created_at: 1,
        status: STATUS_IN_PROGRESS.into(),
        error: None,
        incomplete_details: None,
        model: "m".into(),
        output: Vec::new(),
        previous_response_id: None,
        instructions: None,
        max_output_tokens: None,
        temperature: None,
        top_p: None,
        metadata: None,
        usage: None,
    }
}

fn payload(frame: &(&'static str, String)) -> serde_json::Value {
    serde_json::from_str(&frame.1).unwrap()
}

#[test]
fn sequence_numbers_increase_across_all_frames() {
    let mut seq = EventSeq::new();
    let resp = envelope();
    let item = OutputItem::Message {
        id: "msg_1".into(),
        status: STATUS_IN_PROGRESS.into(),
        role: "assistant",
        content: Vec::new(),
    };
    let part = OutputContent::OutputText {
        text: String::new(),
        annotations: Vec::new(),
    };
    let frames = vec![
        seq.created(&resp),
        seq.in_progress(&resp),
        seq.output_item_added(0, &item),
        seq.content_part_added("msg_1", 0, 0, &part),
        seq.output_text_delta("msg_1", 0, 0, "Hi"),
        seq.output_text_done("msg_1", 0, 0, "Hi"),
        seq.content_part_done("msg_1", 0, 0, &part),
        seq.output_item_done(0, &item),
        seq.completed(&resp),
    ];
    for (i, frame) in frames.iter().enumerate() {
        let p = payload(frame);
        assert_eq!(p["sequence_number"], i as u64, "frame {i}");
        assert_eq!(p["type"], frame.0, "frame {i}");
    }
}

#[test]
fn lifecycle_frames_embed_the_envelope() {
    let mut seq = EventSeq::new();
    let resp = envelope();
    let (name, data) = seq.created(&resp);
    assert_eq!(name, EV_CREATED);
    let p: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(p["response"]["id"], "resp_1");
    assert_eq!(p["response"]["object"], "response");

    let (name, _) = seq.in_progress(&resp);
    assert_eq!(name, EV_IN_PROGRESS);
    let (name, _) = seq.completed(&resp);
    assert_eq!(name, EV_COMPLETED);
    let (name, _) = seq.failed(&resp);
    assert_eq!(name, EV_FAILED);
}

#[test]
fn delta_frames_carry_locator_fields() {
    let mut seq = EventSeq::new();
    let frame = seq.output_text_delta("msg_7", 2, 1, "tok");
    assert_eq!(frame.0, EV_OUTPUT_TEXT_DELTA);
    let p = payload(&frame);
    assert_eq!(p["item_id"], "msg_7");
    assert_eq!(p["output_index"], 2);
    assert_eq!(p["content_index"], 1);
    assert_eq!(p["delta"], "tok");
}

#[test]
fn text_done_carries_full_text() {
    let mut seq = EventSeq::new();
    let frame = seq.output_text_done("msg_7", 0, 0, "full text");
    assert_eq!(frame.0, EV_OUTPUT_TEXT_DONE);
    assert_eq!(payload(&frame)["text"], "full text");
}

#[test]
fn item_frames_embed_the_item() {
    let mut seq = EventSeq::new();
    let item = OutputItem::FunctionCall {
        id: "fc_1".into(),
        status: STATUS_COMPLETED.into(),
        call_id: "call_1".into(),
        name: "f".into(),
        arguments: "{}".into(),
    };
    let added = seq.output_item_added(0, &item);
    assert_eq!(added.0, EV_OUTPUT_ITEM_ADDED);
    assert_eq!(payload(&added)["item"]["type"], "function_call");
    let done = seq.output_item_done(0, &item);
    assert_eq!(done.0, EV_OUTPUT_ITEM_DONE);
    assert_eq!(payload(&done)["item"]["name"], "f");
}

#[test]
fn content_part_frames_embed_the_part() {
    let mut seq = EventSeq::new();
    let part = OutputContent::OutputText {
        text: "abc".into(),
        annotations: Vec::new(),
    };
    let added = seq.content_part_added("msg_1", 0, 0, &part);
    assert_eq!(added.0, EV_CONTENT_PART_ADDED);
    assert_eq!(payload(&added)["part"]["type"], "output_text");
    let done = seq.content_part_done("msg_1", 0, 0, &part);
    assert_eq!(payload(&done)["part"]["text"], "abc");
}
