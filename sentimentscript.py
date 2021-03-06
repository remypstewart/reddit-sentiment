from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import re
import seaborn as sns
from scipy.stats import spearmanr

api = PushshiftAPI()

# I conducted my initial query to the Pushshift Reddit API on a seperate remote server due to the dataset size. This commented out section features my 
# initial preparation that I saved through the imported pickle files. 
submissions = ["fbys2o","fbuz0g","fbpdpq","fboh7q","fboct4","fbm8ln","fbluu8","fbky6u","fbkagj","otpj2a","otod3l","otnxyb","otmrgd","otmdtg","otm7wh","otlj1l","otj52g","pfopb2","pzdq38","pzdhwn","pzcezo","pz94b0","pyyl4l","pyy5ia","pyxvqr","pywucl","pyug5h", "i5ik4o", "j0j8an",
               "pkjtlw","pkj3nd","pkiwzt","pkiotd","pki53t","pkg7gx","pkdl6x","pkd6vy","pkcq8e","oq93qn","oq8yyz","oq8p4x","oq8jua","oq8fxd", "oq8b0t","oq6sd5", "oq6rlr", "oq6c14","ms8w18", "ms851a", "ms7or8", "ms7m7r", "ms7jb8", "ms5xaf", "ms5sw4", "ms5kt9", "ms3pxd","ejpmko",
               "npzmyi","npy8se","npxlzs","npxlrj","npw0jz","npu4cq","nplrtt","npkt56","npknct", "hg5reo","hg5j83","hg46q6","hg3ydk","hg2bja","hg1jv8","hg0znv","hfz0yf", "hkr9ph", "etm3qa", "etln6t", "etld44", "etj46p", "etdt4e", "etbxe1", "et4x13", "et4892", "et2yy4", "nhzpq4"
               "powfw1","powe8k","powcn5","pow1v4","pouvv1","pot5qd","pospk8","posaj2","porsnx", "hx8fdz","hx78ci","hx60g7","hx55dn","hx0jmj","hwy6me","hwwqwo","hww9ro", "hwvv1g", "gwkl53", "gwhs3o", "gwhir2", "gwdw56", "evze85", "evyskg", "evx9r8", "evx87s", "evx4qe", "ni6mww",
               "er73ie","er70xt", "er6rj4", "er6gwn", "er65ha", "er5bkz", "er1nmw", "er0pag", "er0kh3","ot1jpl", "oszf0l", "osz52a", "osyhwq", "osuiw6", "ostzbc", "osthrr", "ostbe1", "ossj89","ek1gz3", "ejyu7j", "ejy3hj", "ejxmkm", "ejwgn9", "ejsf5t", "ejrn9h", "ejqjfq", "pme0no",
               "i5ik4o", "i5hebw", "i5c7b6", "i5aok0", "i56a42", "i55sqw", "i55nm3", "i55et2", "i50lv9","f3jtb8", "f3jcqp", "f3huzb", "f3hm2p", "f3hkz6", "f3hgew", "f3gmw0", "f3ghrs", "f3d991", "j17uj8", "j164py", "j14eqq", "j13x2v", "j0w42k", "j0v16x", "j0utn7", "j0mw6w", "kl6wm5",
               "l7i3mg", "l7griq", "l7gofv", "l7gi84", "l7gdk4", "l7frfo", "l7fivg", "l7esnj", "l7cch0","obfkh5", "obdz70", "obbdvh", "obbcv0", "obb820", "obb5r4", "ob9xkk", "ob9teo", "ob97tu", "ni7z2z", "ni7834", "ni6pch", "ni6g76", "ni36ju", "ni2r0m", "ni0ahl", "ilba2v", "k5nqoz",
               "pmhnc8", "pmf6oj", "pmchby", "pmbhqi", "pmbbdd", "pmax3g", "pma9iy", "pm7gvh", "k5rlb1", "k5qxyh", "k5qa8h","k5q7qr", "k5nlz7", "k5myah", "k5m06a", "k5lwhv", "ilg9kb", "ilfsff", "ilew6h", "ilevaq", "ileqw2", "ildcnc", "ilcgyv"]

query = api.search_comments(link_ids = submissions,      
                            subreddit='sanfrancisco',
                            filter = ['author', 'id', 'body', 'link_id', 'score', 'created_utc'])

comments = pd.DataFrame([comment.d_ for comment in query])
comments

# Importing data & preprocessing.
comments1 = pd.read_pickle('./target_comments_1.pkl')
comments2 = pd.read_pickle('./target_comments_2.pkl')
comments = pd.concat([comments1, comments2], ignore_index=True)
comments.drop_duplicates(subset=['body'])
comments['word_count'] = comments['body'].str.findall(r'(\w+)').str.len()
comments = comments[~(comments['word_count'] <= 5)] 
comments['body'] = comments['body'].replace(r'\n',' ', regex=True)
comments['body'] = comments['body'].replace(r'&gt',' ', regex=True)
comments['body'] = comments['body'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
comments

# Sentiment analysis & most polarized comments. 

analyzer = SentimentIntensityAnalyzer()
comments['compound'] = [analyzer.polarity_scores(x)['compound'] for x in comments['body']]
pd.set_option('display.max_colwidth', 600)
text = comments[['body', 'compound']].copy()
text.sort_values(by='compound')[:10]
text.sort_values(by='compound', ascending = False)[:10]

# Distribution & frequency visuals. 
sns.histplot(data=comments, x='compound', bins=10)
comments['compound'].value_counts()
sns.relplot(x="compound", y="score", data=comments)
sentiment = comments[~(comments['score'] < 300)]
sns.relplot(x="compound", y="score", data=sentiment)
sentiment = comments[~(comments['score'] > -50)]
sns.relplot(x="compound", y="score", data=sentiment)

scorecomp = comments[['score', 'compound']].copy()
corrrelation = scorecomp.corr(method="spearman")
print(corrrelation)
