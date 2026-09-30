# Legacy manager priority handoff

Run `drain_priority.py --work ORIGINAL_WORK --recovery RECOVERY_WORK` from an
isolated source checkout, under nohup. Do not update the live source checkout.

The controller stops admission by SIGSTOP of the exact original manager, while
its active child jobs continue. It tracks child identities and descendants,
reads zombie wait status, and applies each existing timeout and original deadline.
After all active jobs finish, it replaces only the stopped manager. No active
training is restarted. The new manager runs the original immutable plan/source.

An explicit `admission.json` makes atomic43 wait until recovery probe-decision
is terminal. A technical recovery failure releases baseline work, not a learned
promotion. Other jobs are unaffected. `priority-handoff.json` records the handoff.

Limitation: the legacy manager holds both GPU leases until draining ends. Guide
may wait for fixed training as well as random pilot. If atomic43 already started,
its existing work is drained too; the controller cannot reorder history. The
controller must remain running while the old manager is stopped. SIGTERM or an
ordinary exception resumes the old manager; SIGKILL/host loss cannot be handled.
On such an interruption, identify the original manager from its exact plan argv
and verify process identity before resuming it with SIGCONT.
