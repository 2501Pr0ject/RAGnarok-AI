# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in ragnarok-ai, please report it responsibly.

### How to Report

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please use **[GitHub Security Advisories](https://github.com/2501Pr0ject/ragnarok-ai/security/advisories/new)** to report vulnerabilities privately.

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

| Action | Timeline |
|--------|----------|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 7 days |
| Fix development | Depends on severity |
| Public disclosure | After fix is released |

### What to Expect

1. **Acknowledgment**: We will confirm receipt of your report within 48 hours
2. **Assessment**: We will investigate and assess the severity
3. **Communication**: We will keep you informed of our progress
4. **Fix**: We will develop and test a fix
5. **Release**: We will release the fix and credit you (unless you prefer anonymity)
6. **Disclosure**: We will publish a security advisory

## Security Best Practices

### For Users

- **Keep ragnarok-ai updated** to the latest version
- **Run Ollama locally** — ragnarok-ai is designed for local-first operation
- **Don't expose services** — Keep Ollama and vector stores on localhost
- **Review configurations** — Check `ragnarok.yaml` for sensitive settings

### For Contributors

- **No hardcoded secrets** — Use environment variables
- **Validate inputs** — Sanitize all user inputs
- **Dependencies** — Keep dependencies updated, review security advisories
- **Code review** — All PRs require review before merge

## Scope

### In Scope

- ragnarok-ai core library (`src/ragnarok_ai/`)
- CLI tool
- Official adapters (Ollama, Qdrant, LangChain)
- Configuration handling
- Data processing pipelines

### Out of Scope

- Third-party dependencies (report to their maintainers)
- Ollama security issues (report to Ollama team)
- Vector store security (report to respective projects)
- User misconfiguration

## Known Security Considerations

### Local-First Design

ragnarok-ai is designed to run 100% locally. This means:

- No data is sent to external APIs by default
- LLM inference happens on your machine via Ollama
- Vector stores run locally

### File System Access

The tool reads and writes files for:

- Knowledge base documents
- Test sets
- Evaluation reports
- Checkpoints

Ensure appropriate file permissions in production environments.

### LLM Prompt Injection

When using LLM-as-judge for metrics (faithfulness, relevance), be aware that:

- Malicious content in documents could attempt prompt injection
- We implement basic sanitization, but no system is foolproof
- Review generated test sets before using in CI/CD

---

Thank you for helping keep ragnarok-ai secure!
