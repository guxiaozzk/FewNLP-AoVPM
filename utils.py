'''
templates and label mappings
'''

t1_fine = {
    'content': [' [CLS] for " ', ' ", the sentence " ', ' " has [MASK] emotion [SEP] '],
    'map': [0, 'a', 1, 'x', 2]
}

t2_fine = {
    'content': [' [CLS] for " ', ' ", the sentence " ', ' " presents a [MASK] sentiment [SEP] '],
    'map': [0, 'a', 1, 'x', 2]
}

# [MASK][P] [A] [P] [X] [P]
# t3_fine = {
#     'content': ['[CLS] [MASK]', '[SEP]'],
#     'map': [0, 'p', 'a', 'p', 'x', 'p', 1]
# }
##for change it to SEntence-levle
# t3_fine = {
#     'content': ['[CLS] How [MASK]  the sentence ', 'has [MASK] emotion [SEP]'],
#     'map': [0, 'p', 'x', 'p', 1]
# }
##it is [MASK] how [MASK]
t3_fine = {
    'content': ['[CLS] the sentence" ', '"has [MASK] emotion [SEP] for','it is [MASK]'],
    'map': [0, 'p', 'x',1,'a', 'p', 2,'p']
}
# t3_fine = {
#     'content': ['[CLS] " ', '"[MASK] [SEP] for','[MASK]'],
#     'map': [0, 'p', 'x',1,'a', 'p', 2,'p']
# }
# t3_fine = {
#     'content': ['[CLS] how [MASK] for " ','  ",the sentence" ', '"has [MASK] emotion [SEP]'],
#     'map': [0, 'p', 'a', 1,'p', 'x', 'p', 2]
# }
template_fine = {
    1: t1_fine,
    2: t2_fine,
    3: t3_fine
}

t1_coarse = {
    'content': [' [CLS] the sentence " ', ' " has [MASK] emotion [SEP] '],
    'map': [0, 'x', 1]
}
##0 :第一词 'x' template 1 第二词
t2_coarse = {
    'content': [' [CLS] the sentence " ', ' " presents a [MASK] sentiment [SEP] '],
    'map': [0, 'x', 1]
}

# [MASK][P] [X] [P]原3
# t3_coarse = {
#     'content': ['[CLS] [MASK]', '[SEP]'],
#     'map': [0, 'p', 'x', 'p', 1]
# }
#
# 改3
# t3_coarse = {
#     'content': ['[CLS]  how [MASK] the sentence ', 'has [MASK] emotion [SEP]'],
#     'map': [0, 'p', 'x', 'p', 1]
# }
# t3_coarse = {
#     'content': ['[CLS] [MASK]', '[MASK] [SEP]'],
#     'map': [0, 'p', 'x', 'p', 1]
# }

t3_coarse = {
    'content': ['[CLS]  how [MASK] the sentence ', 'presents a [MASK] sentiment [SEP]'],
    'map': [0, 'p', 'x', 'p', 1]
}

template_coarse = {
    1: t1_coarse,
    2: t2_coarse,
    3: t3_coarse
}


def twitter(template: int):
    if template == 1:
        label_list = ['bad', 'no', 'good']
        label_map = {'0':"bad", '1': "no", '2': "good"}
    elif template == 2 or template == 3:
        # label_list = ['bad', 'no', 'good']
        # label_map = {'0':"bad", '1': "no", '2': "good"}
        label_list = ['negative', 'neutral', 'positive']
        label_map = {'0':"negative", '1': "neutral", '2': "positive"}
    else:
        raise ValueError("Illegal template")
    return label_list, label_map, template_fine[template]


def masad(template: int):
    if template == 1:
        label_list = ['bad', 'good']
        label_map = {'negative': "bad", 'positive': "good"}
    elif template == 2 or template == 3:
        label_list = ['negative', 'positive']
        label_map = {'negative':"negative", 'positive': "positive"}
    else:
        raise ValueError("Illegal template")
    return label_list, label_map, template_fine[template]


def mvsa(template: int):
    if template == 1:
        label_list = ['bad', 'no', 'good']
        label_map = {'negative':"bad", 'neutral': "no", 'positive': "good"}
    elif template == 2 or template == 3:
        label_list = ['negative', 'neutral', 'positive']
        label_map = {'negative':"negative", 'neutral': "neutral", 'positive': "positive"}
    else:
        raise ValueError("Illegal template")
    return label_list, label_map, template_coarse[template]

def mvsam(template: int):
    if template == 1:
        label_list = ['bad', 'no', 'good']
        label_map = {'negative':"bad", 'neutral': "no", 'positive': "good"}
    elif template == 2 or template == 3:
        # label_list = ['bad', 'no', 'good']
        # label_map = {'negative':"bad", 'neutral': "no", 'positive': "good"}

        label_list = ['negative', 'neutral', 'positive']
        label_map = {'negative':"negative", 'neutral': "neutral", 'positive': "positive"}
    else:
        raise ValueError("Illegal template")
    return label_list, label_map, template_coarse[template]


def tumemo(template: int):
    label_list = ['angry', 'bored', 'calm', 'fear', 'happy', 'love', 'sad']
    label_map = {'Angry': 'angry', 'Bored': 'bored', 'Calm': 'calm', 'Fear': 'fear', 'Happy': 'happy', 'Love': 'love', 'Sad': 'sad'}
    return label_list, label_map, template_coarse[template]


processors = {
    't2015': twitter,
    't2017': twitter,
    'masad': masad,
    'mvsa-s': mvsa,
    'tumemo': tumemo,
    'mvsa-m':mvsam
}
