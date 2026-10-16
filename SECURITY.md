# Security and data handling

The results API is read-only and binds to `127.0.0.1` by default. SQLite is opened in
read-only mode for API requests and snapshot export. The dashboard cannot start training or modify results. Its optional Media demo can send
selected media to a separate local inference service.

This is a local research application. The API has no authentication, authorization, or rate
limiting. Before exposing it on a network, place it behind an authenticated reverse proxy,
configure TLS and CORS for the intended origin, and review the exported result records.
Prefer static hosting for a public demonstration.

Snapshots include experiment notes, run metadata, hyperparameters, library versions, and
artifact paths. Check these records before publishing. Raw datasets and feature caches
are excluded from Git; dataset licences and access terms still apply.

For a vulnerability that could disclose private data or modify results, use the repository's
GitHub private vulnerability reporting if available. Otherwise contact the maintainer
privately through their GitHub profile. Do not include credentials, private datasets, or
exploit details in a public issue.

The opt-in local demo is a separate service on port 8001. Its Media demo view uploads only
after the user presses Analyze. The service caps raw requests at 25 MiB, serializes inference,
deletes temporary files after success/failure, and disables access logs when started by the CLI.
It does not persist uploads or write detection results to SQLite. Keep it on loopback; its CORS
allowlist is not authentication. Trained bundles are local artifacts excluded from Git.
