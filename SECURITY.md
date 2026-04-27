# Cirron SDK Security Policy

This document explains how to report security issues for `cirron-sdk`.

## Reporting a vulnerability

Please report suspected security issues privately.

- **Preferred**: use GitHub's "Report a vulnerability" link in the repository sidebar (or open a [private security advisory](https://github.com/cirron/cirron-sdk/security/advisories/new) directly).
- **Alternative**: email `security@cirron.com`.
- Do **not** open a public issue, pull request, or discussion post for suspected vulnerabilities.

When reporting, please include:

- A description of the issue and the affected version(s) of `cirron-sdk`.
- Steps to reproduce (ideally a minimal proof of concept).
- An assessment of potential impact (data exposure, code execution, denial of service, etc.).
- Your contact information.
- Any specific requests, such as anonymity for you and/or the organization you represent.

## Maintainer commitments

We aim to handle reports quickly and responsibly.

- We will acknowledge receipt within 2 business days.
- We will provide an initial assessment as capacity allows, generally within one to two weeks for non-trivial reports.
- We will share progress updates until resolution.
- Disclosure will be coordinated with the reporter, and credit given in the release notes unless anonymity is requested.

## Disclosure and embargoes

By default, `cirron-sdk` does **not** accept long embargoes. Security reports usually become public once a fix is available and confirmed, alongside a GitHub Security Advisory and a patched release on PyPI.

A short embargo may be considered in exceptional cases (e.g. coordinated downstream patching for a known enterprise deployment), but is not guaranteed. Indefinite embargoes burn maintainer time tracking them and delay protection for users who are already exposed.

## Expectations for reporters

Reporters should understand that:

- Security reports are handled by the Cirron team. We are a small team. Response times reflect that.
- Reports with clear reproducers and a working proof of concept are triaged faster.
- Public exploitation while a fix is in progress shifts the calculus toward faster disclosure, not longer embargoes.

## Supported versions

We backport security fixes to:

- The current minor release of the latest major version.
- The previous major version, for 6 months after the next major ships.

Older versions receive fixes only at the maintainers' discretion. Pin your install (`cirron-sdk>=X.Y,<X+1`) and upgrade promptly to stay supported.

## Scope

In scope:

- Vulnerabilities in `cirron-sdk` source code published on PyPI.
- Vulnerabilities in the SDK's interaction with the local spool / snapshot directories (`./.cirron/`).
- Vulnerabilities in the SDK's network transports (HTTP / kernel event stream) when configured against the Cirron platform.

Out of scope (please report to the upstream project, not us):

- Vulnerabilities in third-party dependencies (`torch`, `tensorflow`, `pandas`, etc.).
- Vulnerabilities in the Cirron platform backend (email `security@cirron.com` separately and mark the subject "platform").
- Issues that require a malicious local user with filesystem write access. The SDK trusts its own spool directory by design.

## Transparency

We document resolved security issues in the project release notes and as GitHub Security Advisories. This helps users understand our process and triage their own exposure.

## Recognition

We thank reporters in the release notes for the fix unless you prefer to remain anonymous. Cirron does not currently run a paid bug bounty program.
