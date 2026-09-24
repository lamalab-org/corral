# Codebook - panel delivery

Tab-separated, one row per delivered record. Responses run from 1 (Disagree) to 5 (Agree). Every item was answered; there are no missing responses.

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

| variable | description |
|---|---|
| `participant_id` | identifier assigned by the panel on delivery |
| `session_id` | identifier assigned when the record was transmitted |
| `recorded_on` | date the record was transmitted |
| `age` | respondent age in years |
| `gender` | 1 = Male, 2 = Female |
| `accuracy` | self-rated response accuracy, 0-100 |
| `country` | collection country; all rows are US |

Rows delivered: 12,965.
