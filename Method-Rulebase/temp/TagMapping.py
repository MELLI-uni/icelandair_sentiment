import pickle
import json

guess_list = ['af', 'afe', 'afm']
pos_tag_list = []
pos_tag_dict = {}

with open('./tags.json', encoding='utf-8') as f:
    tagmap = json.load(f)
f.close()

with open('pos_tag.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
f.close()

for line in lines:
    pos_tag_list.append(line.strip())

for tag in pos_tag_list:
    if tag in guess_list:
        tag = 'ao'
    if tag in tagmap:
        if tagmap[tag].startswith('ao'):
            pos_tag_dict[tag] = 'ADV'

        elif tagmap[tag].startswith('lo'):
            pos_tag_dict[tag] = 'ADJ'

        elif tagmap[tag].startswith('no'):
            pos_tag_dict[tag] = 'NOUN'

        elif tagmap[tag].startswith('so'):
            pos_tag_dict[tag] = 'VERB'

with open('posmap.pickle', 'wb') as handle:
    pickle.dump(pos_tag_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#print(pos_tag_dict)