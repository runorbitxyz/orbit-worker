# Security

The worker runs on other people's machines next to their model servers, so its security matters more than anything
else in this repository. If you find a way for an ORBIT server, a model server, a job or anything else to make the
worker run code, read or send files, leak its worker token or API key, or send anything it promises not to (see the
README), please tell us privately.

**Report it** through GitHub's private vulnerability reporting: the **Security** tab of this repository, then
**Report a vulnerability**. Please don't open a public issue for it.

We'll acknowledge it within a few days, keep you posted while we fix it, and credit you in the release notes if you'd
like. Fixes ship as a new release; the installers always download the latest worker.
