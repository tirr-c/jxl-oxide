# Security Policy

Please report a security bug using [GitHub Security Advisory form][ghsa-form].

[ghsa-form]: https://github.com/tirr-c/jxl-oxide/security/advisories/new

## Supported Versions

In principle, only the latest stable release is supported. Previous stable releases might not get
patches even if they are vulnerable.

## Out of scope

These are not considered a security bug; please open an issue in [the issue tracker][new-issue].
Pull requests are also welcome!

- **Panics and safe aborts.** Users are expected to add panic handler or sandbox the decoder if they
  are using jxl-oxide in production.
- **Unsoundness which isn't triggered by image decoding.**

[new-issue]: https://github.com/tirr-c/jxl-oxide/issues/new

## Using AI tools

Using AI tools to find vulnerabilities is allowed, but before submitting a report, **a human must
verify** if it's really a security bug in scope. **AI agent must not report a bug by itself**
without human intervention.
