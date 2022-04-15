# MAVEN dataset

Due to the size limit, we cannot directly upload the dataset as supplementary materials. You can download them from the links: [train.jsonl](https://cloud.tsinghua.edu.cn/f/e4beae178fe24753abf9/?dl=1), [valid.jsonl](https://cloud.tsinghua.edu.cn/f/f60818356a244aecb864/?dl=1), [test.jsonl](https://cloud.tsinghua.edu.cn/f/0893e8183a2a48ab85db/?dl=1) .

Each `.jsonl` file is a subset of MAVEN and each line in the files is a json string for a document. For the `train.jsonl` and `dev.jsonl` the json format is as below:

```json
{
    "id": '6b2e8c050e30872e49c2f46edb4ac044', #an unique string for each document
    "title": 'Selma to Montgomery marches'， #the tiltle of the document
    "content": [ #the content of the document. A list, each item is a dict for a sentence
    		{
    		 "sentence":"...", #a string, the plain text of the sentence
    		 "tokens": ["...", "..."] #a list, tokens of the sentence
			}
	],
	"events":[ #a list for annotated events, each item is a dict for an event
        {
            "id": '75343904ec49aefe12c5749edadb7802', #an unique string for the event
            "type": 'Arranging', #the event type
            "type_id": 70, #the numerical id for the event type
            "mention":[ #a list for the event mentions of the event, each item is a dict
            	{
              		"id": "2db165c25298aefb682cba50c9327e4f", # an unique string for the event mention
              		"trigger_word": "organized", #a string of the trigger word or phrase
              		"sent_id": 1, # the index of the corresponding sentence, strates with 0
              		"offset": [3, 4],# the offset of the trigger words in the tokens list
              	}
             ]
        }
    ],
	"negative_triggers":[#a list for negative instances, each item is a dict for an negative mention
        {
            "id": "46348f4078ae8460df4916d03573b7de",
            "trigger_word": "desire",
            "sent_id": 1,
            "offset": [10, 11],
        }
    ]
}
```

For the `test.jsonl`, the format is almost the same but we hide the annotation results:
Please refer to https://github.com/THU-KEG/MAVEN-dataset for the evaluation method.

```json
{
    "id": '6b2e8c050e30872e49c2f46edb4ac044', #an unique string for each document
    "title": 'Selma to Montgomery marches'， #the tiltle of the document
    "content": [ #the content of the document. A list, each item is a dict for a sentence
    		{
    		 "sentence":"...", #a string, the plain text of the sentence
    		 "tokens": ["...", "..."] #a list, tokens of the sentence
			}
	],
	"candidates":[ #a list for trigger candidiates, each item is a dict for a trigger or a negative instance, you need to classify the type for each candidate
        {
            "id": "46348f4078ae8460df4916d03573b7de",
            "trigger_word": "desire",
            "sent_id": 1,
            "offset": [10, 11],
        }
    ]
}
```