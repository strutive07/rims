# Rims


## How to run new model experiment


### Run all code in one script.
```bash
# Before you run
# please modify OPENAI_API_BASE, OPENAI_API_KEY in this script and run script.

cd src
sh scripts run_all_scripts.sh
```


### Run each step manualy

```
export OPENAI_API_BASE=
export OPENAI_API_KEY=

MODEL_NAME="blabla"

sh scripts/0_run_indiv.sh $MODEL_NAME
sh scripts/1_run_postprocess_indiv.sh $MODEL_NAME
sh scripts/2_score_indiv.sh $MODEL_NAME
sh scripts/3_run_rims.sh $MODEL_NAME
sh scripts/3_run_sg.sh $MODEL_NAME
sh scripts/4_postprocess_and_score_rims.sh $MODEL_NAME
sh scripts/4_postprocess_and_score_sg.sh $MODEL_NAME
```

If you have problem that is resolved by calling it again like `connection error` on the turn you call LLM, the script is configured so that running it again will resolve the problem.


### Print scores
```bash
python3 print_score.py

cat scores/gsm8K_test_dt.gsm.md
cat scores/MATH-full_dt.math.md
cat scores/ocw_course_dt.ocw.md
```


## LLM and Data scenario


### Step 0. Run cot, pal, p2c. `scripts/0_run_indiv.sh`

LLM is called via async query object for managing data and metadata.
base query object: https://github.com/strutive07/rims/blob/main/src/query_obj/base_query.py

We have 3 individual query strategy, so we override base query object and implement each strategy

- cot: https://github.com/strutive07/rims/blob/main/src/query_obj/cot.py
- p2c: https://github.com/strutive07/rims/blob/main/src/query_obj/p2c.py
- pal: https://github.com/strutive07/rims/blob/main/src/query_obj/pal.py


simple greedy and rims's LLM call also made with base_query object.

- simple_greedy: https://github.com/strutive07/rims/blob/main/src/query_obj/simple_greedy.py
- rims: https://github.com/strutive07/rims/blob/main/src/query_obj/rims.py


Raw output sample: https://github.com/strutive07/rims/blob/main/sample_indiv_output.json


### Step 1. post process llm raw output. `scripts/1_run_postprocess_indiv.sh`

After call llm, we have to parse output from cot, and run generated code for pal and p2c.

https://github.com/strutive07/rims/blob/main/src/postprocess_rawouts.py#L63-L81

In this code, we compare cot, pal, p2c's output. If we fine major answer (two or more methods gave the same answer), we do not run simple greedy and rims in step 3.

Output sample

```json
{
    "question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "cot_solutions": [
        "Answer:\nJanet's ducks lay 16 eggs per day.\nShe eats 3 eggs for breakfast, so she has 16 - 3 = 13 eggs left.\nShe bakes 4 eggs for muffins, so she has 13 - 4 = 9 eggs left.\nShe sells the remaining 9 eggs at the farmers' market.\nEach egg is sold for $2, so she makes 9 x 2 = 18 dollars.\nSo the answer is 18."
    ],
    "pal_solutions": [
        "def solution():\n    \"\"\"Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\"\"\"\n    eggs_laid = 16\n    eggs_eaten = 3\n    eggs_baked = 4\n    eggs_sold = eggs_laid - eggs_eaten - eggs_baked\n    money_made = eggs_sold * 2\n    result = money_made\n    return result"
    ],
    "p2c_solutions": [
        "def solution():\n    '''\n    Create a function solution that returns the answer of the following question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? \n    \n    Let's think step by step.\n    1. Calculate the total number of eggs laid by the ducks in a day: 16 eggs.\n    2. Calculate the number of eggs eaten by Janet for breakfast: 3 eggs.\n    3. Calculate the number of eggs used to bake muffins: 4 eggs.\n    4. Calculate the total number of eggs remaining: 16 - 3 - 4 = 9 eggs.\n    5. Calculate the total amount of money made by selling the remaining eggs: 9 eggs * $2 per egg = $18.\n    6. Return the result: $18.\n    '''\n    # Step 1: Calculate the total number of eggs laid by the ducks in a day\n    total_eggs = 16\n    \n    # Step 2: Calculate the number of eggs eaten by Janet for breakfast\n    eggs_eaten = 3\n    \n    # Step 3: Calculate the number of eggs used to bake muffins\n    eggs_used_for_muffins = 4\n    \n    # Step 4: Calculate the total number of eggs remaining\n    eggs_remaining = total_eggs - eggs_eaten - eggs_used_for_muffins\n    \n    # Step 5: Calculate the total amount of money made by selling the remaining eggs\n    money_made = eggs_remaining * 2\n    \n    # Step 6: Return the result\n    return money_made"
    ],
    "cot_preds": [
        18
    ],
    "pal_preds": [
        18
    ],
    "p2c_preds": [
        18
    ],
    "majvote_answers": [
        18
    ],
    "need_selection": [
        false
    ],
    "dataset_type": "gsm",
    "gt_answer": 18
}
```


### Step 2. calculate cot, pal, p2c scores. `scripts/2_score_indiv.sh`

You can see the score `outputs/*/*/processed_indiv_scored.md`


### Step 3. run simple greedy, rims `scripts/3_run_sg.sh`, `scripts/3_run_rims.sh`

In step 1, we computed data instances that do not have a major answer. We use simple greedy and rims to perform model selection and reflection.

Prompt and output samples

- Simple greedy

Prompt template: https://github.com/strutive07/rims/blob/main/src/query_obj/simple_greedy_prompts.yaml

```json
{
    "SimpleGreedyQueryObject": {
        "contents": [
            "(A) can correctly answer the math problem."
        ],
        "query_message": [
            {
                "role": "system",
                "content": "You are a helpful assistant that can identify the correct answer to the math problem."
            },
            {
                "role": "user",
                "content": "There are three choices to the same math problem. One uses natural language to answer the question, while the other two uses Python program to answer it. One directly attempts to answer the problem with coding the solution, while the last one will plan first and then write a code to answer it. Three of them can correctly answer the math problem. You need to identify which choice can correctly answer the math problem. Here is one example how to do it,\n\nMath problem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n\n(A)\nAnswer:\nOlivia had 23 dollars.\n5 bagels for 3 dollars each will be 5 * 3 = 15 dollars.\nSo she has 23 - 15 = 8 dollars left.\nSo the answer is 8.\n\n(B)\ndef solution():\n    money_initial = 23\n    bagels = 5\n    bagel_cost = 3\n    money_spent = bagels + bagel_cost\n    money_left = money_initial - money_spent\n    result = money_left\n    return result\n\n(C)\ndef solution():\n    '''\n    Create a function solution that returns the answer of the following question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? \n        \n    Let's think step by step.\n    1. Calculate the total cost of the bagels.\n    2. Subtract the total cost from Olivia's original amount.\n    3. Return the remaining amount.\n    '''\n    # Step 1: Calculate the total cost of the bagels\n    bagel_cost = 3\n    num_bagels = 5\n    total_cost = bagel_cost * num_bagels\n    \n    # Step 2: Subtract the total cost from Olivia's original amount\n    original_amount = 23\n    remaining_amount = original_amount - total_cost - num_bagels\n    \n    # Step 3: Return the remaining amount\n    return remaining_amount\n\nWhich of the above three choices can correctly answer the math problem?\n\n(A) can correctly answer the math problem. Because (B) adds the number of bagels to the cost of each bagel instead of multiplying them, and (C) also fails by wrongly subtracting num_bagels from original_amount.\n\nNow it's your turn. Here is another math problem and three choices.\nMath Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n\n(A)\nAnswer:\nMichael started with 58 golf balls.\nThen after losing 23 on tuesday, he had 58 -23 = 35.\nAfter losing 2 more, he had 35 + 2 = 37 golf balls.\nSo the answer is 37.\n\n(B)\ndef solution():\n    golf_balls_initial = 58\n    golf_balls_lost_tuesday = 23\n    golf_balls_lost_wednesday = 2\n    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday + golf_balls_lost_wednesday\n    result = golf_balls_left\n    return result\n\n(C)\ndef solution():\n    '''\n    Create a function solution that returns the answer of the following question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? \n\n    Let's think step by step.\n    1. Initialize the variable `golf_balls` with the initial number of golf balls.\n    2. Subtract the number of golf balls lost on Tuesday and Wednesday.\n    3. Return the remaining number of golf balls. \n    '''\n    # Step 1: Initialize the variable `golf_balls` with the initial number of golf balls\n    golf_balls = 58\n    \n    # Step 2: Subtract the number of golf balls lost on Tuesday (23) and Wednesday (2)\n    golf_balls -= 23  # golf balls lost on Tuesday\n    golf_balls -= 2   # golf balls lost on Wednesday\n    \n    # Step 3: Return the remaining number of golf balls\n    return golf_balls\n\nWhich of the above three choices can correctly answer the math problem?\n"
            },
            {
                "role": "assistant",
                "content": "(C) can correctly answer the math problem. Because the others wrongly add up 'lost' golf balls to the total count of the balls."
            },
            {
                "role": "user",
                "content": "Math problem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\n\n(A)\nAnswer:\nThere were originally 9 computers.\nFor each of 4 days from monday to thursday, 5 more computers were added, that is, total 5 computers added for 4 days.\nSo there are 9 + 5 = 14 computers now.\nSo the answer is 14.\n\n(B)\ndef solution():\n    computers_initial = 9\n    computers_per_day = 5\n    num_days = 4\n    computers_added = computers_per_day * num_days\n    computers_total = computers_initial + computers_added\n    result = computers_total\n    return result\n\n(C)\ndef solution():\n    '''\n    Create a function solution that returns the answer of the following question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? \n        \n    Let's think step by step.\n    1. Calculate the total number of computers installed from Monday to Thursday: 5 computers/day * 3 days = 15\n    2. Add the initial 9 computers to the total: 9 + 15 = 24\n    3. There are now 24 computers in the server room.\n    '''\n    return 9 + 5 * 3\n\nWhich of the above three choices can correctly answer the math problem?\n"
            },
            {
                "role": "assistant",
                "content": "(B) can correctly answer the math problem. Because (A) fails to add up computers added each day, and (C) wrongly figured the number of days for computers to be added as 3 days."
            },
            {
                "role": "user",
                "content": "Math problem: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\n\n(A) \nAnswer:\nJosh bought the house for $80,000.\nThen he put in $50,000 in repairs.\nThe value of the house increased by 150%.\nSo the new value of the house is 150% of the original value.\n150% of $80,000 is 1.5 x $80,000 = $120,000.\nSo the new value of the house is $120,000.\nJosh sold the house for $120,000.\nHe bought the house for $80,000 and spent $50,000 in repairs.\nSo his total cost was $80,000 + $50,000 = $130,000.\nHis profit is the difference between the selling price and the cost.\n$120,000 - $130,000 = -$10,000.\nSo Josh actually lost $10,000.\nSo the answer is -$10,000.\n\n(B) \ndef solution():\n    \"\"\"Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\"\"\"\n    house_price = 80000\n    repairs = 50000\n    total_cost = house_price + repairs\n    value_after_repair = total_cost * 3.5  # 150% increase\n    profit = value_after_repair - total_cost\n    result = profit\n    return result\n\n(C) \nHere is the Python function that solves the problem:\n\n\n\nWhich of the above three choices can correctly answer the math problem?\n"
            }
        ],
        "meta": {
            "method_obj": "SimpleGreedyQueryObject",
            "dataset_type": "gsm",
            "query_kwargs": {
                "backbone": "meta-llama/Meta-Llama-3-8B-Instruct",
                "temperature": 0,
                "n": 1,
                "seed": 777,
                "max_tokens": 512,
                "stop": "\n\n\n"
            },
            "gt_answer": 70000
        }
    }
}
```

- rims

Prompt template: https://github.com/strutive07/rims/tree/main/src/query_obj/rims_prompts

```json
{
    "RimsQueryObject": {
        "contents": [
            "I will attempt to solve the problem using the Chain-of-Thought (cot) method.\n\n`Method`: Chain-of-Thought (cot)\n`Attempt 1`:\nLet's think step by step.\n1. Josh buys a house for $80,000 and then puts in $50,000 in repairs.\n2. The value of the house increases by 150%.\n3. To find the new value of the house, we need to calculate 150% of the original value.\n4. 150% of $80,000 is $120,000.\n5. The total amount spent by Josh is $80,000 (initial cost) + $50,000 (repairs) = $130,000.\n6. The profit is the difference between the new value and the total amount spent.\n7. Profit = New value - Total amount spent\n8. Profit = $120,000 - $130,000 = -$10,000\n\n`Answer 1`: -10,000\n`Evaluation`: Wrong\n`Mistakes`: The mistake is in the calculation of the new value of the house. The value of the house increases by 150%, but the calculation is incorrect. The correct calculation is 150% of $80,000 is $120,000, but the new value is not $120,000, it's the original value plus the increase, which is $80,000 + $80,000 = $160,000.\n`Hint for a better Method choice`: Using a program-aided approach can help avoid human errors in arithmetic and provide a precise result.\n`Workaround Method`: Program-aided Language Modeling (pal)\n`Attempt 2`:\ndef solution():\n    initial_cost = 80000\n    repairs = 50000\n    increase_percentage = 150\n    new_value = initial_cost * (1 + increase_percentage/100)\n\n    total_amount_spent = initial_cost + repairs\n    profit = new_value - total_amount_spent\n\n    return profit\n\nprint(solution())\n\n`Answer 2`: 30000"
        ],
        "query_message": [
            {
                "role": "user",
                "content": "You are now solving math word problems. You brilliantly detects the errors in the wrong solution and find `Workaround Method` to correct the solution. The methods you are taking are as follows. Each has its strength and weakness:\n\n- Chain of Thought (cot): Solving problem with writing steps of reasoning in a natural language. Might help correct understanding of the problem but this could be weaker in precise computation.\n- Program-aided Language Modeling (pal): Using python language to reason and obtain an accurate answer to the given question, but this could be weaker in understanding the problem.\n- Plan-and-then-Code (p2c): When a question seems requiring amount of steps to reach the answer, write plans first for what to compute and write a python code to it for solving the problem. However if planning goes wrong, the code will also be wrong. If any steps of planning provided before programming, then it will be considered as Plan-and-then-Code.\n\nFollowings are the examples of correcting the wrong solutions with a `Workaround Method` based on diagnosis (`Mistakes`) and `Hint for a better Method choice`.\n\n\n\n\n`Question`: Jerry is making cherry syrup. He needs 500 cherries per quart of syrup. It takes him 2 hours to pick 300 cherries and 3 hours to make the syrup. How long will it take him to make 9 quarts of syrup?\n`Method`: Program-aided Language Modeling (pal)\n`Attempt 1`:\ndef solution():\n    cherries_per_quart = 500\n    total_quarts = 9\n    total_cherries_needed = cherries_per_quart * total_quarts\n\n    cherries_per_2_hours = 300\n    time_per_300_cherries = 2  # hours\n\n    # Time to pick cherries for 9 quarts\n    # We find how many 2-hour sessions we need by dividing the total cherries needed by the cherries picked in 2 hours.\n    # We use the ceiling to ensure we pick enough cherries even if the total isn't a multiple of 300.\n    time_to_pick_per_quart = time_per_300_cherries / cherries_per_2_hours * cherries_per_quart\n    total_time_to_pick = time_to_pick_per_quart * total_quarts\n\n    syrup_making_time = 3  # hours for each quart\n    total_time_to_make_syrup = syrup_making_time * total_quarts\n\n    # Total time to make 9 quarts of syrup is the sum of picking time and making time\n    total_time = total_time_to_pick + total_time_to_make_syrup\n    return total_time\n\n`Answer 1`: 57.0\n`Evaluation`: Wrong\n`Mistakes`: The error in the first attempt came from incorrect calculation of the time taken to pick cherries. The attempt wrongly assumes that the syrup making time is multiplied by the number of quarts, which is not stated in the problem. The correct approach should consider the cherry picking time proportional to the number of cherries but treat the syrup making as a fixed duration independent of the quantity.\n`Hint for a better Method choice`: Taking the problem step by step with a plan to address each part of the process (picking and making) separately could avoid compounding errors and lead to a correct solution. Writing out the plan before implementing the code will help verify the logic before execution.\n`Workaround Method`: Plan-and-then-Code (p2c)\n`Attempt 2`: \ndef solution():\n    '''\n    Create a function solution that returns the answer of the following question: Jerry is making cherry syrup. He needs 500 cherries per quart of syrup. It takes him 2 hours to pick 300 cherries and 3 hours to make the syrup. How long will it take him to make 9 quarts of syrup? \n        \n    Let's think step by step.\n    1. Calculate the total number of cherries needed for 9 quarts of syrup.\n    2. Calculate the total number of cherries picked per hour.\n    3. Calculate the total number of hours it will take to pick all the cherries needed.\n    4. Calculate the total time to make 9 quarts of syrup including picking and making the syrup. \n    5. Return the total time.\n    '''\n    # Step 1: Calculate the total number of cherries needed for 9 quarts of syrup\n    cherries_needed = 500 * 9\n    \n    # Step 2: Calculate the total number of cherries picked per hour\n    cherries_per_hour = 300 / 2\n    \n    # Step 3: Calculate the total number of hours it will take to pick all the cherries needed\n    hours_to_pick_cherries = cherries_needed / cherries_per_hour\n    \n    # Step 4: Calculate the total time to make 9 quarts of syrup including picking and making the syrup\n    total_time = hours_to_pick_cherries + 3\n    \n    # Step 5: Return the total time\n    return total_time\n`Answer 2`: 33.0\n`Evaluation`: Correct\n\n\n\n`Question`: A man drives 60 mph for 3 hours.  How fast would he have to drive over the next 2 hours to get an average speed of 70 mph?\n`Method`: Chain-of-Thought (cot)\n`Attempt 1`: To solve this problem, we need to work out the total distance the man would need to travel in 5 hours to have an average speed of 70 mph, and then determine how fast he would have to drive in the last 2 hours to cover the remaining distance.\n\nStep 1: Calculate the target total distance for an average speed of 70 mph over 5 hours.\nTotal distance = Average speed × Total time\nTotal distance = 70 mph × 5 hours = 350 miles\n\nStep 2: Calculate the distance he covered in the first 3 hours driving at 60 mph.\nDistance covered in 3 hours = Speed × Time\nDistance covered = 60 mph × 3 hours = 180 miles\n\nStep 3: Calculate the remaining distance he still needs to cover in the next 2 hours to reach the target total distance of 350 miles.\nRemaining distance = Target total distance - Distance already covered\nRemaining distance = 350 miles - 180 miles = 170 miles\n\nStep 4: Calculate the speed he needs to maintain in the next 2 hours to cover the remaining distance of 170 miles.\nSpeed required = Remaining distance / Time left\nSpeed required = 170 miles / 2 hours = 85 mph\n\nSo, to achieve an average speed of 70 mph over the total journey of 5 hours, the man would have to drive at a speed of 85 mph over the next 2 hours.\n`Answer 1`: 2.0\n`Evaluation`: Wrong\n`Mistakes`: The diagnosis of the first attempt does not seem to have an error in the thought process—the mistake lies in the final answer given (\"2.0\") which does not match the correct answer calculated (\"85 mph\").\n`Hint for a better Method choice`: Using a programming approach can help avoid simple calculation errors and ensures that the computed values are correctly reported as the final answer.\n`Workaround Method`: Plan-and-then-Code (p2c)\n`Attempt 2`: \ndef solution():\n    '''\n    Create a function solution that returns the answer of the following question: A man drives 60 mph for 3 hours.  How fast would he have to drive over the next 2 hours to get an average speed of 70 mph?\n    \n    Let's think step by step.\n    1. Calculate the total distance covered in the first 3 hours (60 mph * 3 hours).\n    2. Calculate the distance needed in the next 2 hours to have an average speed of 70 mph over 5 hours.\n    3. Calculate the speed needed for the next 2 hours to cover the remaining distance in 2 hours.\n    4. Return the speed needed for the next 2 hours.\n    '''\n    # Step 1: Calculate the total distance covered in the first 3 hours (60 mph * 3 hours)\n    distance_covered_3_hours = 60 * 3\n    \n    # Step 2: Calculate the distance needed in the next 2 hours to have an average speed of 70 mph over 5 hours\n    average_speed = 70\n    total_time = 5\n    total_distance = average_speed * total_time\n    \n    distance_needed_next_2_hours = total_distance - distance_covered_3_hours\n    \n    # Step 3: Calculate the speed needed for the next 2 hours to cover the remaining distance in 2 hours\n    speed_needed_next_2_hours = distance_needed_next_2_hours / 2\n    \n    # Step 4: Return the speed needed for the next 2 hours\n    return speed_needed_next_2_hours\n`Answer 2`: 85.0\n`Evaluation`: Correct\n\n\n\n`Question`: There are enough provisions in a castle to feed 300 people for 90 days. After 30 days, 100 people leave the castle. How many more days are left until all the food runs out?\n`Method`: Chain-of-Thought (cot)\n`Attempt 1`: Answer:\nOriginally, there are enough provisions to feed 300 people for 90 days.\n\nStep 1: Calculate the total provisions in terms of \"person-days,\" which is the number of days that a given amount of food can feed one person. This is done by multiplying the number of people by the number of days the food can last.\n\nTotal provisions = 300 people * 90 days = 27000 person-days\n\nStep 2: Determine how many person-days have been used after 30 days with 300 people.\n\nProvisions used in 30 days = 300 people * 30 days = 9000 person-days\n\nStep 3: Subtract the provisions used from the total provisions to find out how many person-days are left.\n\nProvisions left = Total provisions - Provisions used\nProvisions left = 27000 person-days - 9000 person-days\nProvisions left = 18000 person-days\n\nStep 4: Calculate how many days the remaining food will last for the reduced number of people (200 people) after 100 people have left.\n\nDays left for food = Provisions left / Number of people remaining\nDays left for food = 18000 person-days / 200 people\nDays left for food = 90 days\n\nTherefore, after 30 days have passed and 100 people leave, the castle has enough provisions to last for another 90 days for the remaining 200 people.\n`Answer 1`: 200.0\n`Evaluation`: Wrong\n`Mistakes`: The error in the first attempt is in the final step of calculation where the answer is incorrectly represented as 200 days. The correct calculation of days left for food with the remaining 200 people should be 18000 person-days divided by 200 people, which gives 90 days, not 200.\n`Hint for a better Method choice`: For complex calculations involving several steps, a program-aided approach can help avoid human errors in arithmetic and provide a precise result.\n`Workaround Method`: Program-aided Language Modeling (pal)\n`Attempt 2`:\ndef solution():\n    initial_people = 300\n    provisions_for_days = 90\n    days_passed = 30\n    people_left = 100\n\n    # Calculate total provisions for 300 people for 90 days\n    total_provisions = initial_people * provisions_for_days\n\n    # Calculate provisions consumed in 30 days\n    provisions_consumed = initial_people * days_passed\n\n    # Calculate remaining provisions after 30 days\n    remaining_provisions = total_provisions - provisions_consumed\n\n    # Calculate number of people remaining after 100 people leave\n    remaining_people = initial_people - people_left\n\n    # Calculate how many days provisions will now last with the remaining 200 people\n    days_remaining = remaining_provisions // remaining_people\n\n    return days_remaining\n\n# Call the function and print the result\nprint(solution())\n\n`Answer 2`: 90.0\n`Evaluation`: Correct\n\n\n\nNow, try the `Question` below following the same procedure as above. Try the question with the choice of your `Method`, and evaluate the `Answer`. If your `Attempt` is considered wrong, identify the `Mistakes` and reason to take `Workaround Method` by writing `Hint for a better Method choice`. Based on it, make a correct reattempt.\n\n`Question`: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
            }
        ],
        "meta": {
            "method_obj": "RimsQueryObject",
            "dataset_type": "not given",
            "query_kwargs": {
                "backbone": "meta-llama/Meta-Llama-3-8B-Instruct",
                "temperature": 0,
                "n": 1,
                "seed": 777,
                "max_tokens": 2048,
                "stop": [
                    "\n`Evaluation`: Correct",
                    "`Evaluation`: Correct",
                    "Evaluation: Correct"
                ]
            },
            "gt_answer": 70000
        }
    }
}
```


### Step 4. Post prcessing simple greedy and rims `scripts/4_postprocess_and_score_sg.sh`, `scripts/4_postprocess_and_score_rims.sh`

Parse selected strategy and answer.

sample output

```json
{
    "question": "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?",
    "cot_solutions": [
        " Answer:\nJosh bought the house for $80,000 and put in $50,000 in repairs.\nSo he spent a total of $80,000 + $50,000 = $130,000.\nThe value of the house increased by 150%.\nSo the increase in value is 150% of $130,000.\nTo find 150% of $130,000, we multiply $130,000 by 1.5 (since 150% is the same as 1.5 times).\nSo the increase in value is $130,000 \\* 1.5 = $195,000.\nTherefore, the value of the house is now $130,000 + $195,000 = $325,000.\nJosh bought the house for $80,000 and sold it for $325,000.\nSo he made a profit of $325,000 - $80,000 = $245,000.\nSo the answer is $245,000."
    ],
    "pal_solutions": [
        " def solution():\n    \"\"\"Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\"\"\"\n    house_cost = 80000\n    repair_cost = 50000\n    total_cost = house_cost + repair_cost\n    increase_percentage = 150\n    increase_amount = total_cost * (increase_percentage / 100)\n    house_value_after = total_cost + increase_amount\n    profit = house_value_after - total_cost\n    result = profit\n    return result"
    ],
    "p2c_solutions": [
        " def solution():\n    '''\n    Create a function solution that returns the answer of the following question: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\n\n    Let's think step by step.\n     1. Calculate the total cost of the house and repairs.\n    2. Calculate the increased value of the house.\n    3. Calculate the profit.\n    4. Return the profit.\n    '''\n    # Step 1: Calculate the total cost of the house and repairs\n    total_cost = 80000 + 50000\n\n    # Step 2: Calculate the increased value of the house\n    increased_value = total_cost * 1.5\n\n    # Step 3: Calculate the profit\n    profit = increased_value - total_cost\n\n    # Step 4: Return the profit\n    return profit"
    ],
    "cot_preds": [
        245000
    ],
    "pal_preds": [
        195000
    ],
    "p2c_preds": [
        65000
    ],
    "majvote_answers": [
        null
    ],
    "need_selection": [
        true
    ],
    "dataset_type": "gsm",
    "gt_answer": 70000,
    "rims_selected": "cot",
    "rims_solution": "Josh bought the house for $80,000 and spent $50,000 on repairs, so the total cost was $80,000 + $50,000 = $130,000.\n\nThe value of the house increased by 150%, so the new value of the house is $130,000 + (150% of $130,000) = $130,000 + $195,000 = $325,000.\n\nThe profit Josh made is the new value of the house minus the total cost, which is $325,000 - $130,000 = $205,000.",
    "rims_answer": 205000,
    "rims_summary": {
        "good_solution": "Josh bought the house for $80,000 and spent $50,000 on repairs, so the total cost was $80,000 + $50,000 = $130,000.\n\nThe value of the house increased by 150%, so the new value of the house is $130,000 + (150% of $130,000) = $130,000 + $195,000 = $325,000.\n\nThe profit Josh made is the new value of the house minus the total cost, which is $325,000 - $130,000 = $205,000.",
        "good_ans": 205000,
        "good_method": "cot",
        "bad_solutions": [],
        "bad_ans": [],
        "bad_method": [],
        "mistakes": [],
        "hint": [],
        "did_reflect": 0
    },
    "error": false,
    "raw_text": null,
    "exception": null
}
```