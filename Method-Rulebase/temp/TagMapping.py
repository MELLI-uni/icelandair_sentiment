import pickle
import json

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

counter = 0
non_matching = 0
not_matching = []

for tag in pos_tag_list:
    if tag in tagmap:
        pos_tag_dict[tag] = tagmap[tag]
        counter += 1
    else:
        not_matching.append(tag)
        non_matching += 1

print("matching: ", counter)
print("not matching: ", non_matching)

for items in not_matching:
    print(items)