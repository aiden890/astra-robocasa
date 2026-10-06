# Parallel rollout execution

Each rollout uses an independent simulator process and a distinct output directory. New slots use private Spark2 containers, each capped at 2 CPU cores and 12 GiB memory. Existing workers and their loaded code are preserved. The model remains gpt-6-astra using the same subscription and unchanged observation/action contract.

The coordinator adopts the original Panda run, starts GR1, measures a 150-second two-session window, then adds one Panda and one GR1 session and measures another 150-second window. It stops scaling if a worker ends. Four sessions continue until native success or 1,800 steps per episode, with automatic 20fps publication.

The private .runtime/parallel-plan.json records run IDs, seeds, containers, and the adopted PID. .runtime/parallel-report.json records completed model calls, action repeats, response latency, and aggregate steps per wall-clock second. Failed startup attempts remain in their original directories and are never overwritten. The initial GR1 startup failure occurred before any model call or native action; a separate native reset/render/step probe passed after the duplicate option was removed.

The measurement is a short operational throughput comparison, not a paired performance or success-rate experiment. Different robots, phases, and model-selected action repeats can change the number of steps per call. Initial reset overhead is included in each parallel window. Four is the tested configuration, not a proved host or subscription maximum. Subscription usage is shared; the official documentation does not specify a fixed per-account concurrency ceiling: https://learn.chatgpt.com/docs/pricing .

No API credentials, model fallback, additional purchased credits, or altered robot action bounds are used.
