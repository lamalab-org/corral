# Codebook - online personality survey

Tab-separated, one row per respondent. Item responses are ratings on a five-point scale: 1 = Disagree, 3 = Neutral, 5 = Agree. **0 = missed.**

Items are taken from published self-report personality scales and were presented to respondents in a single block.

| item | text |
|---|---|
| HSNS1 | I can become entirely absorbed in thinking about my personal affairs, my health, my cares or my relations to others. |
| HSNS2 | My feelings are easily hurt by ridicule or the slighting remarks of others. |
| HSNS3 | When I enter a room I often become self conscious and feel that the eyes of others are upon me. |
| HSNS4 | I dislike sharing the credit of an achievement with others. |
| HSNS5 | I feel that I have enough on my hands without worrying about other people's troubles. |
| HSNS6 | I feel that I am temperamentally different from most people. |
| HSNS7 | I often interpret the remarks of others in a personal way. |
| HSNS8 | I easily become wrapped up in my own interests and forget the existence of others. |
| HSNS9 | I dislike being with a group unless I know that I am appreciated by at least one of those present. |
| HSNS10 | I am secretly "put out" or annoyed when other people come to me with their troubles, asking me for my time and sympathy. |
| DDM1 | I tend to manipulate others to get my way. |
| DDM2 | I have used deceit or lied to get my way. |
| DDM3 | I have used flattery to get my way. |
| DDM4 | I tend to exploit others towards my own end. |
| DDP1 | I tend to lack remorse. |
| DDP2 | I tend to not be too concerned with morality or the morality of my actions. |
| DDP3 | I tend to be callous or insensitive. |
| DDP4 | I tend to be cynical. |
| DDN1 | I tend to want others to admire me. |
| DDN2 | I tend to want others to pay attention to me. |
| DDN3 | I tend to seek prestige or status. |
| DDN4 | I tend to expect special favors from others. |

| variable | description |
|---|---|
| `age` | entered as free text |
| `gender` | 1 = Male, 2 = Female, 3 = Other, 0 = missed |
| `accuracy` | self-rated accuracy of own responses, 0-100 |
| `country` | ISO country code |

Rows: 39,000. Countries: US (24,000), GB (6,200), CA (3,800), AU (3,000), DE (830), IN (660), BR (510).
