"""Background flush thread — stub for SDK-11.

Responsibilities (per spec §3.3):
- batch scope closes + marks
- write to ``./.cirron/spool/``
- push via event stream (platform) or HTTP (external)
"""
