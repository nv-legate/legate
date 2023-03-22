# Basic Hello, World Application

The code for this example can be found in the [library file](../hello/hello.py) and [example](hello-world.py).

## Single, auto task

Generally auto tasks should be preferred that automatically
partition and parallelize task launches.
In the hello world example, only a single scalar argument
is added and the task is enqueued with `execute`:

```
task = user_context.create_auto_task(HelloOpCode.HELLO_WORLD)
task.add_scalar_arg(message, types.string)
task.execute()
```

In this case, the cost heuristic in the runtime will notice
that the task is inexpensive and launch a single instance.

## Manual task with explicit launch domain

It is possibly to manually specify the launch domain for a task,
overriding the internal heuristics.

```
launch_domain = Rect(lo=[0], hi=[n], exclusive=True)
task = user_context.create_manual_task(
    HelloOpCode.HELLO_WORLD, launch_domain=launch_domain
)
task.add_scalar_arg(message, types.string)
task.execute()
```

Now `n` replica tasks will be launched. In this case,
the `Rect` launch domain is linear, but multi-dimensional domains
are also possible.
