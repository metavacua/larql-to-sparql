use crate::sparql::parse_pairs;

/// Run a SPARQL query against query.wikidata.org and return (subject,object) label pairs.
/// Wikidata requires a descriptive User-Agent.
pub fn fetch_pairs(query: &str) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("larql-knowledge/0.1 (https://github.com/metavacua/larql-to-sparql; metavacua@gmail.com)")
        .build()?;
    let resp = client
        .get("https://query.wikidata.org/sparql")
        .header("Accept", "application/sparql-results+json")
        .query(&[("query", query), ("format", "json")])
        .send()?
        .error_for_status()?;
    let body = resp.text()?;
    Ok(parse_pairs(&body, "sLabel", "oLabel")?)
}
