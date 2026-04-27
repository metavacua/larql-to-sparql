use larql_core::engine::templates::{PromptTemplate, TemplateRegistry};

#[test]
fn test_empty_registry() {
    let reg = TemplateRegistry::new();
    assert!(reg.all().is_empty());
    assert!(reg.relations().is_empty());
    assert!(!reg.has("anything"));
    assert!(reg.get("anything").is_none());
}

#[test]
fn test_register_and_get() {
    let mut reg = TemplateRegistry::new();
    reg.register(PromptTemplate {
        relation: "capital-of".to_string(),
        template: "The capital of {subject} is".to_string(),
        reverse_template: None,
        multi_token: true,
        stop_tokens: vec!['.', '\n'],
    });

    assert!(reg.has("capital-of"));
    let tmpl = reg.get("capital-of").unwrap();
    assert_eq!(tmpl.template, "The capital of {subject} is");
    assert!(tmpl.multi_token);
}

#[test]
fn test_format_subject() {
    let tmpl = PromptTemplate {
        relation: "capital-of".to_string(),
        template: "The capital of {subject} is".to_string(),
        reverse_template: Some("The country with capital {object} is".to_string()),
        multi_token: true,
        stop_tokens: vec![],
    };

    assert_eq!(tmpl.format("France"), "The capital of France is");
    assert_eq!(
        tmpl.format_reverse("Paris"),
        Some("The country with capital Paris is".to_string())
    );
}

#[test]
fn test_format_no_reverse() {
    let tmpl = PromptTemplate {
        relation: "r".to_string(),
        template: "{subject} test".to_string(),
        reverse_template: None,
        multi_token: false,
        stop_tokens: vec![],
    };

    assert!(tmpl.format_reverse("anything").is_none());
}

#[test]
fn test_json_roundtrip() {
    let mut reg = TemplateRegistry::new();
    reg.register(PromptTemplate {
        relation: "capital-of".to_string(),
        template: "The capital of {subject} is".to_string(),
        reverse_template: None,
        multi_token: true,
        stop_tokens: vec!['.', '\n'],
    });
    reg.register(PromptTemplate {
        relation: "currency".to_string(),
        template: "The currency of {subject} is".to_string(),
        reverse_template: None,
        multi_token: true,
        stop_tokens: vec![],
    });

    let json = reg.to_json_value();
    let restored = TemplateRegistry::from_json_value(&json);

    assert!(restored.has("capital-of"));
    assert!(restored.has("currency"));
    assert_eq!(restored.all().len(), 2);

    let cap = restored.get("capital-of").unwrap();
    assert_eq!(cap.template, "The capital of {subject} is");
    assert!(cap.multi_token);
}

#[test]
fn test_load_from_example_file() {
    let contents = std::fs::read_to_string("../../examples/templates.json").unwrap();
    let value: serde_json::Value = serde_json::from_str(&contents).unwrap();
    let reg = TemplateRegistry::from_json_value(&value);

    assert!(reg.has("capital-of"));
    assert!(reg.has("currency"));
    assert!(reg.has("birthplace"));
    assert!(reg.has("located-in"));
    assert_eq!(reg.all().len(), 12);
}
