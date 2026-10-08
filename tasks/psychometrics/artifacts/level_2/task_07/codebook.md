# Codebook - behavioural validity

The training and holdout files are tab-separated and already joined by `participant_id`. Ratings run from 1 (Disagree) to 5 (Agree); 0 denotes a missing item response. The behavioural outcome is continuous.

| variable | description |
|---|---|
| `participant_id` | participant identifier, unique within each file |
| `gender` | 1 = Group 1, 2 = Group 2 |
| `behavior` | independently measured behavioural outcome; higher values indicate more of the behaviour |

The training file is `data.csv`; `holdout.csv` was collected separately and has no training/holdout indicator in the data.

| item | text |
|---|---|
| `HSNS1` | I can become entirely absorbed in thinking about my personal affairs, my health, my cares or my relations to others. |
| `HSNS2` | My feelings are easily hurt by ridicule or the slighting remarks of others. |
| `HSNS3` | When I enter a room I often become self conscious and feel that the eyes of others are upon me. |
| `HSNS4` | I dislike sharing the credit of an achievement with others. |
| `HSNS5` | I feel that I have enough on my hands without worrying about other people's troubles. |
| `HSNS6` | I feel that I am temperamentally different from most people. |
| `HSNS7` | I often interpret the remarks of others in a personal way. |
| `HSNS8` | I easily become wrapped up in my own interests and forget the existence of others. |
| `HSNS9` | I dislike being with a group unless I know that I am appreciated by at least one of those present. |
| `HSNS10` | I am secretly "put out" or annoyed when other people come to me with their troubles, asking me for my time and sympathy. |
