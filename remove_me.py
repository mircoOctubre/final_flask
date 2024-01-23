# se establecen los labels y se escriben en el archivo ./annotations/label_map.pbtxt
labels = [
    {'name':'cuchillo', 'id':1},
    {'name':'pistola', 'id':2}
    ]
#se creara el archivo
with open("./annotations/label_map.pbtxt", 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
