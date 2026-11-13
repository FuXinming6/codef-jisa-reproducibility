# End-to-end ablation integrity audit

Result: PASS. Validated 240 rows = 3 families × 20 seeds × 4 arms.
Paired graph/action/constraint fingerprints: 60 bundles; all four arms match within each bundle.
Full-arm reproduction: all 60 objectives and initial action portfolios match the saved main benchmark; 180 attacker-profile records match for risk, observed path, continuation, foothold, and response actions.
Single working-tree source SHA-256: `39cb9b5cf5df7216336cf34cf6ac729614dbfc58d385656e00386e700ba04580`.
Single git HEAD: `4c40ffdfcc9e5dd78af0a802a23724fbff8ab717` (working tree was modified; source hash is the run-level code fingerprint).
Budget total, node/action counts, phase-one path cap, and response-pool base size validated for every paired bundle.

This integrity check establishes consistent instances and reproduction of the full arm. It does not establish model validity, global optimality, or real-network external validity.
