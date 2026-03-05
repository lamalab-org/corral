python 1_slope.py \
    --models="claude-4.5,gpt-4o,gpt-oss-120b" \
    --qa_type_for_ordering=reasoning_qa \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=claude \
    --order_direction=ascending \
    --verbosity_strategy=average \
    --task_type_strategy=both \
    --level_strategy=default_map \
    --metric=pass_at_k \
    --k_value=5 \
    --output_filename=claude_reasoningqa_avgverbosity_bothtasktype_leveldefault_passat5.pdf


python 2_slope.py \
    --models="claude-4.5,gpt-4o,gpt-oss-120b" \
    --qa_type_for_ordering=reasoning_qa \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=claude \
    --order_direction=ascending \
    --verbosity_strategy=average \
    --task_type_strategy=both \
    --level_strategy=default_map \
    --metric=pass_at_k \
    --k_value=5 \
    --agent_type_strategy=average \
    --output_filename=agent_average_level_default_qa_reasoning_order_claude_verbosity_average_task_both.pdf


python 2_slope.py \
    --models="claude-4.5,gpt-4o,gpt-oss-120b" \
    --qa_type_for_ordering=qa \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=claude \
    --order_direction=ascending \
    --verbosity_strategy=average \
    --task_type_strategy=both \
    --level_strategy=default_map \
    --metric=pass_at_k \
    --k_value=5 \
    --agent_type_strategy=average \
    --output_filename=agent_average_level_default_qa_knowledge_order_claude_verbosity_average_task_both.pdf


python 3_gap_plot.py \
    --qa_type_for_ordering=reasoning_qa \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=claude \
    --order_direction=ascending \
    --verbosity_strategy=average \
    --task_type_strategy=both \
    --level_strategy=default_map \
    --metric=pass_at_k \
    --k_value=5 \
    --qa_type_for_ordering=reasoning_qa \
    --output_filename=plots/gaps_reasoning_1.pdf


python 3_gap_plot.py \
    --qa_type_for_ordering=reasoning_qa \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=claude \
    --order_direction=ascending \
    --verbosity_strategy=average \
    --task_type_strategy=tasks \
    --level_strategy=default_map \
    --metric=pass_at_k \
    --k_value=5 \
    --qa_type_for_ordering=reasoning_qa \
    --output_filename=plots/gaps_reasoning_2.pdf


python 3_gap_plot.py \
    --qa_type_for_ordering=reasoning_qa \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=claude \
    --order_direction=ascending \
    --verbosity_strategy=average \
    --task_type_strategy=tasks \
    --level_strategy=all \
    --qa_type_for_ordering=reasoning_qa \
    --metric=pass_at_k \
    --k_value=5 \
    --output_filename=plots/gaps_reasoning_3.pdf


python 3_gap_plot.py \
    --qa_type_for_ordering=reasoning_qa \
    --ordering_strategy=average_qa \
    --order_direction=ascending \
    --verbosity_strategy=average \
    --task_type_strategy=both \
    --level_strategy=default_map \
    --metric=pass_at_k \
    --k_value=5 \
    --qa_type_for_ordering=reasoning_qa \
    --output_filename=plots/gaps_reasoning_4.pdf


python 3_gap_plot.py \
    --qa_type_for_ordering=qa \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=claude \
    --order_direction=ascending \
    --verbosity_strategy=average \
    --task_type_strategy=both \
    --level_strategy=default_map \
    --metric=pass_at_k \
    --k_value=5 \
    --output_filename=plots/gaps_knowledge_1.pdf



python 3_gap_plot.py \
    --qa_type_for_ordering=qa \
    --ordering_strategy=average_qa \
    --model_for_ordering=claude \
    --order_direction=ascending \
    --verbosity_strategy=average \
    --task_type_strategy=both \
    --level_strategy=default_map \
    --metric=pass_at_k \
    --k_value=5 \
    --output_filename=plots/gaps_knowledge_2.pdf
