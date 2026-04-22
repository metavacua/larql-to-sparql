//! # HTTP status expert
//!
//! Lookup for standard IANA HTTP response codes. Only canonical reason phrases
//! are returned — no editorial prose.
//!
//! ## Ops
//!
//! - `lookup {code: 100..=599} → {code, reason, category} | null`
//!
//! `category` is one of `"1xx"`, `"2xx"`, `"3xx"`, `"4xx"`, `"5xx"`.

use expert_interface::{arg_u64, expert_exports, json, Value};

expert_exports!(
    id = "http_status",
    tier = 1,
    description = "HTTP status code lookup: IANA canonical reason phrases and category",
    version = "0.2.0",
    ops = ["lookup"],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "lookup" => {
            let code = arg_u64(args, "code")? as u16;
            let reason = reason(code)?;
            Some(json!({
                "code": code,
                "reason": reason,
                "category": category(code),
            }))
        }
        _ => None,
    }
}

fn reason(code: u16) -> Option<&'static str> {
    match code {
        100 => Some("Continue"),
        101 => Some("Switching Protocols"),
        102 => Some("Processing"),
        103 => Some("Early Hints"),
        200 => Some("OK"),
        201 => Some("Created"),
        202 => Some("Accepted"),
        203 => Some("Non-Authoritative Information"),
        204 => Some("No Content"),
        205 => Some("Reset Content"),
        206 => Some("Partial Content"),
        207 => Some("Multi-Status"),
        208 => Some("Already Reported"),
        226 => Some("IM Used"),
        300 => Some("Multiple Choices"),
        301 => Some("Moved Permanently"),
        302 => Some("Found"),
        303 => Some("See Other"),
        304 => Some("Not Modified"),
        305 => Some("Use Proxy"),
        307 => Some("Temporary Redirect"),
        308 => Some("Permanent Redirect"),
        400 => Some("Bad Request"),
        401 => Some("Unauthorized"),
        402 => Some("Payment Required"),
        403 => Some("Forbidden"),
        404 => Some("Not Found"),
        405 => Some("Method Not Allowed"),
        406 => Some("Not Acceptable"),
        407 => Some("Proxy Authentication Required"),
        408 => Some("Request Timeout"),
        409 => Some("Conflict"),
        410 => Some("Gone"),
        411 => Some("Length Required"),
        412 => Some("Precondition Failed"),
        413 => Some("Content Too Large"),
        414 => Some("URI Too Long"),
        415 => Some("Unsupported Media Type"),
        416 => Some("Range Not Satisfiable"),
        417 => Some("Expectation Failed"),
        418 => Some("I'm a Teapot"),
        421 => Some("Misdirected Request"),
        422 => Some("Unprocessable Content"),
        423 => Some("Locked"),
        424 => Some("Failed Dependency"),
        425 => Some("Too Early"),
        426 => Some("Upgrade Required"),
        428 => Some("Precondition Required"),
        429 => Some("Too Many Requests"),
        431 => Some("Request Header Fields Too Large"),
        451 => Some("Unavailable For Legal Reasons"),
        500 => Some("Internal Server Error"),
        501 => Some("Not Implemented"),
        502 => Some("Bad Gateway"),
        503 => Some("Service Unavailable"),
        504 => Some("Gateway Timeout"),
        505 => Some("HTTP Version Not Supported"),
        506 => Some("Variant Also Negotiates"),
        507 => Some("Insufficient Storage"),
        508 => Some("Loop Detected"),
        510 => Some("Not Extended"),
        511 => Some("Network Authentication Required"),
        _ => None,
    }
}

fn category(code: u16) -> &'static str {
    match code {
        100..=199 => "1xx",
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _ => "unknown",
    }
}
