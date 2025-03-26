from chembench.evaluate import ChemBenchmark


def group_by_meaningful_keywords(benchmark: ChemBenchmark):
    # Dictionary to store all topics and their grouped tasks
    all_grouped_tasks = {}

    for topic in benchmark.registry.get_all_topics():
        if topic == "chemical_preference":
            continue
        questions = benchmark.registry.get_topic(topic)

        # Dictionary to store tasks grouped by keywords
        grouped_tasks = {}

        for task in questions.tasks:
            name = task._name
            parts = name.split("_")

            # Work backwards through parts to find the first non-digit, non-Roman numeral
            keyword = ""
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i]
                # Check if part is a digit
                if part.isdigit():
                    continue

                # Check if part is a Roman numeral (basic check)
                if all(c in "ivxlcdm" for c in part.lower()):
                    continue

                if any(c.isnumeric() for c in part):
                    continue

                # Found a non-digit, non-Roman numeral part
                keyword = "_".join(parts[: i + 1])
                break

            # If we didn't find a keyword, use the whole name
            if not keyword:
                keyword = "others"

            # Add task to the appropriate group
            if keyword not in grouped_tasks:
                grouped_tasks[keyword] = []
            grouped_tasks[keyword].append(task)

        # Move groups with less than 5 tasks to "others"
        small_groups = [
            k for k, tasks in grouped_tasks.items() if len(tasks) < 5 and k != "others"
        ]

        # Create "others" group if it doesn't exist
        if small_groups and "others" not in grouped_tasks:
            grouped_tasks["others"] = []

        # Move tasks from small groups to "others"
        for group in small_groups:
            grouped_tasks["others"].extend(grouped_tasks[group])
            del grouped_tasks[group]

        # Add to the overall dictionary with UUIDs for each task
        topic_tasks = {}
        for keyword, tasks in grouped_tasks.items():
            topic_tasks[keyword] = [task._uuid for task in tasks]

        all_grouped_tasks[topic] = topic_tasks

    return all_grouped_tasks
