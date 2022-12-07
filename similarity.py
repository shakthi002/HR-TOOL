import textdistance as td



def calc_similarity(rs,j):
    '''
    To Calculate the similarity measures of resume and job description
    :param jd:
    :param resume:
    :return float:
    '''
    jc = td.jaccard.similarity(rs,j)
    s = td.sorensen_dice.similarity(rs, j)
    c = td.cosine.similarity(rs, j)
    o = td.overlap.normalized_similarity(rs, j)
    total = (jc + s + c + o) / 4

    return round(total*100,2)
