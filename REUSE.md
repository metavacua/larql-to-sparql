<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright 2026 Ian Douglas Lawrence Norman McLean

With attribution to Chris Hay for LARQL:
https://github.com/chrishayuk/larql

This documentation is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
https://creativecommons.org/licenses/by-sa/4.0/
-->

# REUSE Compliance

This document explains how the larql-to-sparql project maintains REUSE compliance and how to ensure your contributions follow these standards.

## What is REUSE?

[REUSE Software](https://reuse.software/) is a specification that provides a simple and robust way to declare copyright and license information in software projects. It uses SPDX identifiers in file headers to make licensing unambiguous and machine-readable.

## Licensing Structure

The larql-to-sparql project uses a dual-license approach:

### Source Code: Apache 2.0
All source code files (`.rs`, `.py`, build scripts, etc.) are licensed under **Apache License 2.0**:

- ✅ Can be used commercially
- ✅ Can be modified
- ✅ Can be distributed
- ⚠️ Must include license and copyright notice
- ⚠️ State changes made to the code

**SPDX Identifier:** `Apache-2.0`

### Documentation: Creative Commons BY-SA 4.0
All documentation created in the larql-to-sparql fork (README files, guides, specifications, etc.) are licensed under **Creative Commons Attribution-ShareAlike 4.0 International**:

- ✅ Can be used and adapted
- ✅ Must attribute the original author (Ian Douglas Lawrence Norman McLean)
- ✅ Must attribute LARQL (Chris Hay)
- ✅ Derivative works must use the same license
- 📄 CC BY-SA 4.0 applies specifically to documentation created in the fork

**SPDX Identifier:** `CC-BY-SA-4.0`

## SPDX Headers

### For Source Code Files (Rust, Python, Bash, etc.)

**Rust (.rs files):**
```rust
// SPDX-License-Identifier: Apache-2.0

//! Module description here.
```

**Python (.py files):**
```python
# SPDX-License-Identifier: Apache-2.0

"""Module description here."""
```

**Bash/Shell scripts:**
```bash
#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

# Script description here
```

**Build files (build.rs):**
```rust
// SPDX-License-Identifier: Apache-2.0
```

### For Documentation Files (Markdown, etc.)

**Markdown (.md files):**
```markdown
<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright 2026 Ian Douglas Lawrence Norman McLean

With attribution to Chris Hay for LARQL:
https://github.com/chrishayuk/larql

This documentation is licensed under the Creative Commons Attribution-ShareAlike 4.0 License.
-->

# Document Title

Content here...
```

### For Configuration Files (YAML, TOML, JSON, etc.)

**YAML (.yml, .yaml files):**
```yaml
# SPDX-License-Identifier: Apache-2.0
# or CC-BY-SA-4.0 if documentation

---
# Configuration content here
```

**TOML (.toml files):**
```toml
# SPDX-License-Identifier: Apache-2.0
```

**JSON (.json files):**
```json
{
  "__license": "SPDX-License-Identifier: Apache-2.0",
  "content": {}
}
```

## CI/CD Compliance Checks

### Automated REUSE Validation

The project runs automated REUSE compliance checks on every push and pull request via GitHub Actions (`.github/workflows/reuse-compliance.yml`). The workflow:

1. Checks that all files have proper SPDX license identifiers
2. Validates SPDX identifier format
3. Confirms license files exist for declared licenses
4. Fails the CI build if any files are non-compliant

### Running REUSE Checks Locally

Before committing, you can validate REUSE compliance on your local machine:

```bash
# Install REUSE tool (requires Python 3.7+)
pip install reuse

# Check compliance
reuse lint

# Get detailed report
reuse lint --verbose

# Check specific file
reuse lint --file <filename>
```

### Example Output

**Compliant project:**
```
REUSE Compliance Check
✓ Compliance certificate will be awarded to this project.
```

**Non-compliant project (example):**
```
REUSE Compliance Check
✗ Compliance certificate cannot be awarded to this project.

Missing copyright and licensing information:
- crates/my-module/src/new_file.rs
```

## Troubleshooting REUSE Compliance Issues

### Missing SPDX Header

**Error:** `Non-compliant file: src/file.rs` (no SPDX identifier)

**Fix:** Add SPDX header to the top of the file (after shebang if present):
```rust
// SPDX-License-Identifier: Apache-2.0
```

### Invalid SPDX Identifier

**Error:** `Invalid SPDX expression in file.rs`

**Fix:** Use valid SPDX identifiers:
- ✅ `Apache-2.0` (correct)
- ✅ `CC-BY-SA-4.0` (correct)
- ❌ `Apache License 2.0` (invalid format)
- ❌ `CC-BY-SA-4` (incomplete version)

### License File Missing

**Error:** `License file not found for Apache-2.0`

**Fix:** Ensure `/home/user/larql-to-sparql/LICENSE` exists with the full Apache 2.0 license text.

### Documentation License Mismatch

**Error:** New markdown file lacks CC-BY-SA-4.0 header

**Fix:** Add proper header to documentation:
```markdown
<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright 2026 Ian Douglas Lawrence Norman McLean

With attribution to Chris Hay for LARQL:
https://github.com/chrishayuk/larql

This documentation is licensed under the Creative Commons Attribution-ShareAlike 4.0 License.
-->
```

## File Categorization

### Always Apache 2.0
- `.rs` files (Rust source)
- `.py` files (Python source)
- `build.rs` scripts
- `Cargo.toml` files
- `Makefile` target commands

### Always CC-BY-SA-4.0
- New `.md` files (documentation, guides)
- `CHANGELOG.md` (per-crate changelogs)
- `CONTRIBUTING.md` (contribution guidelines)
- Specifications and architecture docs

### Depends on Origin
- Original LARQL files: Apache 2.0
- Files from other open-source projects: Respect original license
- Configuration files (GitHub Actions, linting configs): Apache 2.0

## Best Practices

1. **Add headers early:** Include SPDX headers when creating new files, not as an afterthought.

2. **Be consistent:** Use the same license for the same file type across the project.

3. **Check before committing:** Run `reuse lint` locally before pushing to avoid CI failures.

4. **Update headers for major changes:** If you significantly refactor a file, update any relevant copyright information in documentation.

5. **Preserve original licenses:** Don't change the license of files from the original LARQL project.

6. **Document exceptions:** If a file must use a different license, document why in a comment.

## References

- **REUSE Specification:** https://reuse.software/spec/
- **SPDX License List:** https://spdx.org/licenses/
- **Apache 2.0 License:** https://opensource.org/licenses/Apache-2.0
- **CC BY-SA 4.0 License:** https://creativecommons.org/licenses/by-sa/4.0/
- **LARQL Repository:** https://github.com/chrishayuk/larql

## Attribution

This project is a fork of **LARQL** (Lazarus Query Language), originally created by Chris Hay.

Original repository: https://github.com/chrishayuk/larql

The larql-to-sparql fork maintains Apache 2.0 licensing for all source code while adding CC BY-SA 4.0 licensed documentation.
