"""
This is a script in Python that imports recent posts from Reddit - 
specifically from auto-immune disease focused subreddits - and analyzes the 
posts for a variety of features including treatments discussed, top keywords, 
and contributor characteristics, such as top authors, ages, etc.
"""

#Import statements

import sys
import praw #PRAW is a Python Reddit API wrapper that streamlines the process
            #of importing Reddit posts
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words as wordlist
from datetime import datetime
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import texttable as tt

id = sys.argv[1]
secret = sys.argv[2]
agent = sys.argv[3]

reddit = praw.Reddit(client_id=id,
                     client_secret=secret,
                     user_agent=agent)

wordList = set(word.lower() for word in wordlist.words())
stopWords = set(stopwords.words('english'))

#Treatment dictionaries are included for 4 disease states to facilitate
#analysis; the lists are not comprehensive, but cover several major and/or 
#in-development treatments used in the US for 4 auto-immune diseases, 
#including MS, diabetes (focused on type 1), psoriasis and Crohn's disease

#Multiple Sclerosis treatment dictionary
MSTreatmentDict = {'Copaxone':('copaxone',),
                 'Ocrevus':('ocrevus','ocrelizumab'), 
                 'Plegridy':('plegridy','peginterferon'),
                 'Tysabri':('tysabri','natalizumab'),
                 'Tecfidera':('tecfidera','dimethyl fumarate'),
                 'Gilenya':('gilenya','fingolimod'),
                 'Aubagio':('aubagio','teriflunomide'),
                 'Avonex':('avonex',),
                 'Rebif':('rebif',),
                 'Lemtrada':('lemtrada','alemtuzumab'),
                 'Betaseron':('betaseron',),
                 'Zinbryta':('zinbryta','daclizumab'),
                 'Glatopa':('glatopa',),
                 'Glatiramer Acetate':('glatiramer',),
                 'Interferon':('interferon',),
                 'Novantrone':('novantrone','mitoxantrone'),
                 'Cladribine':('mavenclad','cladribine'),
                 'Rituxan':('rituxan','rituximab'),
                 'ofatumumab':('ofatumumab','arzerra'),
                 'ozanimod':('ozanimod',)}

#Diabetes treatment dictionary
DiaTreatmentDict = {'Lantus':('lantus',),
                 'Levemir':('levemir',), 
                 'Toujeo':('toujeo',),
                 'Tresiba':('tresiba',),
                 'Basaglar':('basaglar',),
                 'Metformin':('metformin',),
                 'Symlin':('symlin','pramlintide'),
                 'Omnipod':('omnipod',),
                 'Afrezza':('afrezza',),
                 'Novolog':('novolog',),
                 'NovoRapid':('novorapid',),
                 'Novolin':('novolin',),
                 'Humalog':('humalog',),
                 'Humulin':('humulin',),
                 'Apidra':('apidra',),
                 'sotagliflozin':('sotagliflozin',)}

#Psoriasis treatment dictionary
PsTreatmentDict = {'Humira':('humira','adalimumab','cyltezo','amjevita'),
                 'Enbrel':('enbrel','etanercept'), 
                 'Taltz':('taltz','ixekizumab'),
                 'Stelara':('stelara','ustekinumab'),
                 'Remicade':('remicade','infliximab', 'inflectra'),
                 'Cosentyx':('cosentyx','secukinumab'),
                 'Otezla':('otezla','apremilast'),
                 'Cimzia':('cimzia','certolizumab'),
                 'Siliq':('siliq','brodalumab'),
                 'Cyclosporine':('cyclosporine ',),
                 'Methotrexate':('methotrexate','trexall','mtx'),
                 'Soriatane':('acitretin','soriatane')}

#Crohn's Disease treatment dictionary
CDtreatmentDict = {'Humira':('humira','adalimumab'),
                 'Stelara':('stelara','ustekinumab'),
                 'Remicade':('remicade','infliximab','inflectra','renflexis'),
                 'Entyvio':('entyvio','vedolizumab'),
                 'Tysabri':('tysabri','natalizumab'),
                 'Cimzia':('cimzia','certolizumab'),
                 'Azathioprine':('azathioprine','imuran','azasan'),
                 'Methotrexate':('methotrexate','trexall','mtx'),
                 'Mercaptopurine':('mercaptopurine','purinethol','purixan')}

def threads(n=20):
    """
    Takes as input a number of new Reddit submissions (threads) to import
    with 20 as default. Captures both submission/post and its comments.
    Produces as output a list of lists. Each list contains internally 
    assigned thread and post ids, Reddit-assigned ids, author names, 
    author flair, dates, the content of the submissions, and these same 
    categories for comments
    """
    threadz = []
    thread_count = 0
    try:
        for submission in subreddit.new(limit=n):
            submission.comments.replace_more(limit=None)
            date = datetime.utcfromtimestamp(submission.created_utc)
            thread_count +=1
            comment_count = 1
            title = submission.title
            sub_id = submission.id
            try: #if author deletes comment there is no author name for 
                 #comment deleted this try-except block avoids an 
                 #AttributeError by skipping over the comment
                author = submission.author.name
            except:
                print('Attribute error')
                continue
            author_flair = submission.author_flair_text
            text = submission.selftext
            #All of the information captured into variables is stored
            #in a list; in later functions, information is selectively 
            #extracted from this list as needed
            x = [thread_count, comment_count, date.strftime('%b %d, %Y'), 
                 sub_id, author, author_flair, title, text]
            threadz.append(x)
            all_comments = submission.comments.list()
            #this captures all the comments for the current thread
            #then loops through each comment to extract & store relevant info
            for comment in all_comments:
                date = datetime.utcfromtimestamp(comment.created_utc)
                author = str(comment.author)
                author_flair = comment.author_flair_text
                text = comment.body
                comment_count +=1
                comment_id = comment.id
                y = [thread_count, comment_count, date.strftime('%b %d, %Y'), 
                     comment_id, author, author_flair, title, text]
                threadz.append(y)
    except Exception as e: 
        print("General error:",e)
        print("Reducing threads:", n)
        threads(n-1) #if the above author exception fails, loop recursively 
                     #backwards through threads until no errors are raised
    return threadz

def redditText(threadz, author=None):
    """
    Converts text from posts pulled from Reddit into a list of words.
    If an author is specified, returns word list for only that author.
    """
    #Post contents are stored in the 8th "slot" for each post or comment
    #pulled in by the threads function
    if author == None:
        text = [''.join(w[7]) for w in threadz]
    else:
        #Usernames of authors are stored in the 5th slot by the threads func
        text = [''.join(w[7]) for w in threadz if w[4] == author]
    text = str(text)
    cleanTxt = wordsClean(text)
    return cleanTxt
    
def wordsClean(text):
    """
    Takes input of string of text and turns it into a list of words with 
    most punctuation removed; used in Rx and keywords searches
    """
    wordscl = []
    tokens = nltk.word_tokenize(text)
    words = [w.lower() for w in tokens if w is not None]
    for word in words:
        w1 = re.split(r'\\+n|[/\'-]|\|',word) #gets rid of newline chars
                                                #and a few other chars
        for w in w1:
            wordscl.append(w)
    for pos, word in enumerate(wordscl):
        w = word.strip('\'\"\\,.-\\\\/\*!%():;[]`&\+#$^?><’') 
                                #removes some non-alphabetic chars
        wordscl[pos] = w        #from beginning and end of words
                                #and stores the cleaned word in the old 
                                #word's position
    
    wordscl = [w for w in wordscl if w is not '' and not w.isdigit()]
    #Must check again for "empty" words which may have been created from 
    #split and strip loops
    return wordscl

def uncommonWords(words):
    """
    Creates a list of words *not* in English wordlist (NLTK) & not stopwords 
    to use as comparison list for treatments, since treatment names 
    are not in dictionary or stopwords list; this reduces the 
    number of loops and comparisons needed in subsequent functions.
    (Unfortunately, UK spellings don't seem to be included in wordlist).
    This takes a little longer to run than other functions. 
    Can be run for full Reddit corpus or for individual Reddit authors
    """
    redditWords = []
    words = set(w for w in words)
    #Could also be restricted to just alphabetic words using the w.isalpha()
    #method, but want to keep hyphenated words and this would exclude them
               
    uncommonWords = words - wordList - stopWords
    
    for word in uncommonWords:  #Basic "sledgehammer" check for plurals: 
        try:                    #subtracts final letter from each word in Reddit
                                #corpus and check to see if that word is in 
                                #imported word list, b/c word list does not 
                                #contain (any?) plurals
            if word[-1] is not 's' or word[0:-1] not in wordList: 
                    redditWords.append(word)
        except:
            continue
    
    return sorted(redditWords)

def redditRx(words):
    """
    Takes as input list output from uncommonWords function. Compares words
    to words in the appropriate treatment dictionary defined above.
    Vowels are removed for comparison as vowels are the source of many 
    misspellings. An additional comparison is performed by removing 
    1 consonant from each word in uncommonwords to catch misspellings with
    just one additional consonant, e.g. "Techfidera" (cf. "Tecfidera").
    This is meant to capture a non-exhaustive but large percentage of 
    misspellings without manually inputting a list of misspellings. Can
    also be run for individual authors. (I also thought of comparing words 
    by percentage matching and found the pre-existing function SequenceMatcher 
    to do the comparison, but I haven't gotten it to be fully functional yet)
    """
    barewords = []
    barewords1 = defaultdict(list)
    RxListNoVowels = []
    RxListNoVowels1 = defaultdict(list)
    vowels = ['a','e','i','o','u']
    words = list(set(words))
    rxDictNew = defaultdict(list)

    #This loop creates a new list of words variations with vowels removed
    #and each letter removed once to match misspellings in vowels and 
    #some misspellings with consonants (some redundancies arise and 
    #efficiency and scope could be improved, but a lot is captured)
    for word in words:
        bareword = word
        for vowel in vowels:
            bareword = bareword.replace(vowel,"") #removes vowels from words
        barewords.append((bareword, word))          
        xtraclist = list(extraConsonant(word)) #Calls function below
        for x in xtraclist:                     #which generates a list
            barewords.append((x,word))          #of variants for the current
                                                #word with each of its letters
                                                #removed once
    
    #The Index method, which sorts words by their letters (in this case
    #with vowels or one letter removed) is from the NLTK book ch. 5; it is 
    #used again below for the list of treatments
    barewords1 = nltk.Index((''.join(sorted(w)), x) for (w, x) in barewords)
    
    #A list of treatments from the treatment dictionary with and without
    #vowels is also created as a comparator for the words in Reddit posts
    #processed above
    for k,v in treatmentDict.items():
        for rx in v:
            RxNoVowels = rx
            for vowel in vowels:
                RxNoVowels = RxNoVowels.replace(vowel,"")
            RxListNoVowels.append((rx, k))
            RxListNoVowels.append((RxNoVowels, k))
        
    RxListNoVowels1 = nltk.Index((''.join(sorted(w)),x) for (w,x) 
    in RxListNoVowels)
        
    #In this last loop, the two lists are compared: the list of treatments
    #(with and without vowels) and the list of uncommon words from Reddit 
    #posts (without vowels and each consonant removed). When there's a match
    #it's added to a new default dictionary to later turn into a bar chart
    for Rx1,Rx2 in RxListNoVowels1.items():
        for value in barewords1[Rx1]:
            if value:
                if value not in rxDictNew[Rx2[0]]:
                    rxDictNew[Rx2[0]].append(value)

    return rxDictNew

def extraConsonant(word):
    """
    Takes as input a word and generates a list (when called) of variations
    of the word with one letter removed. This is an example of a generator.
    """
    if len(word) <= 1:
        yield word
    else:
        for i in range(len(word)):
            i+=1
            yield word[:i-1] + word[i:]

def treatmentCount(words, treatm):
    """
    Takes as input 1) the list of treatments (output of redditRx function) and 
    2) the full Reddit corpus text. Returns a dictionary with counts of how 
    many times each treatment appears in the full text.
    """
    treatmentNum = defaultdict(int)
    
    #count number of times a treatment name, including its variants and
    #misspellings, appear in the corpus
    for word in words:
        for k,v in treatm.items():
            if word in treatm[k]:
                treatmentNum[k]+=1
    
    names = list(treatmentNum.keys())
    counts = list(treatmentNum.values())
    plt.bar(names,counts)
    plt.suptitle('Treatments mentioned in %r Subreddit' % subreddit.display_name)
    plt.xticks(rotation=65)
    plt.show()

def datesPlot(threadz, author=None):
    """
    Creates frequency plot of dates of Reddit posts and comments. Can also 
    be created for individual author activity.
    """
    allDates = defaultdict(int)
    
    if author == None:
        for item in threadz:
            #Dates are stored in the 3rd "slot" for each post or comment
            #pulled in by the threads function
            date = item[2]
            allDates[date]+=1
    else:
        for item in threadz:
            if item[4] == author:                
                date = item[2]
                allDates[date]+=1
        
    names = list(allDates.keys())
    counts = list(allDates.values())    
    plt.bar(names,counts)
    plt.suptitle('Recent %r Subreddit posts by date' % subreddit.display_name)
    plt.xticks(rotation=65)
    plt.show() 
    
def keywords(words, n=20):
    """
    Takes as input the output of the redditText function, for either the full
    corpus of Reddit text or for an individual author, and returns the top
    20 keywords. Excludes most common words - function words, pronouns, 
    common verbs - i.e. "stopwords" then also excludes remaining one-letter 
    words because they are almost always reduced from contractions, 
    e.g. "don't" > do n't > do n t
    """
    stopw = list(stopWords)
    wordFD = nltk.FreqDist(w for w in words if w not in stopw and len(w)>1)
    return wordFD.most_common(n)

def authorCount(threadz): 
    """
    Creates a new dictionary of Reddit authors based on threads function.
    Keys are usernames of Reddit authors and values are number of posts
    written by that author.
    """
    authors = defaultdict(int)
    for item in threadz:
        #Usernames of authors are stored in the 5th "slot" for each post or 
        #comment pulled in by the threads function
        author = item[4]
        authors[author]+=1
    return authors

def topAuthors(authors,n=20):
    """
    Determines top 20 authors based on number of posts/comments.
    Takes as input dictionary of authors: the output of authorCount 
    function. And n = length of list output (here fixed at n=20).
    (A different way to count top authors could be to count 
    number of total words written by the author, not number of posts)
    """
    topauth = []
    for key, value in authors.items():
        topauth.append((value,key))
    topauth = sorted(topauth, reverse=True)
    topauth = topauth[:n]
    topauth = [b for (a,b) in topauth]
    return topauth

class redditor(object):
    """
    This class gathers and organizes information about individual authors
    active in Reddit communities. It draws on many of the functions
    above, functions which are used to analyze data at both the author
    level and the overall community level (full corpus). Not all of the 
    methods written for the class are used in the analysis output at the 
    end, but this could be modified to create author-level reports
    or separate author analyses.
    """
    def __init__(self, author, flair=None, posts=0, text=None, rxs=None,
                 rx=None, gen=None, dx=None):
        self.author = author #username of Redditor or Reddit author
        self.flair = flair or None #user provided personal information
        self.posts = posts #number of posts by author in time period
        self.text = text or None #list of text author has written in recent posts
        self.rxs = rxs or None #treatments mentioned in author posts
        self.rx = rx or None #treatment author is currently taking
        self.gen = gen or None #gender/sex
        self.dx = dx or None #date of disease diagnosis
        #flair, which can contain age, sex (male or female), treatment and 
        #diagnosis information is only available occasionally, when the 
        #author chooses to include it
    
    def getInfo(self,thrds):
        """
        Some communities use "flair" which is a place to store personal 
        information or the type of info in an email signature; it is up to 
        individual Reddit users whether to include Flair. The Psoriasis 
        community does not appear to use Flair at all.
        """
        #Author Flair is stored in the 6th "slot" for each post or 
        #comment pulled in by the threads function
        self.flair = next((item[5] for item in thrds if item[4]==self.author), 
                          'No author info available')
        if self.flair == None:
            self.flair = 'No author info available'
            
    def getFlair(self):
        return self.flair
    
    def getPosts(self,thrds):
        """
        Counts the number of posts written by the author
        """
        for item in thrds:
            if item[4] == self.author:
                self.posts+=1
        return self.posts
    
    def getText(self, thrds):
        """
        Converts text from the author's posts into a list of words 
        """
        self.text = redditText(thrds, self.author)
        return self.text
    
    def getRxsText(self):
        """
        Must run findText first before getting Rx dictionary.
        Gathers a list of all relevant treatments mentioned in author's posts.
        """
        if self.text == None:
            print("Must get author text before Rx list (use method: findText)")
            return
        unc = uncommonWords(self.text)
        self.rxs = redditRx(unc)
        return self.rxs
    
    def timeline(self,thrds):
        """
        Plots dates of author posts on a bar chart
        """
        return datesPlot(thrds, self.author)
    
    def getKeywords(self):
        """
        Must run findText first before getting keywords.
        Produces a list of top 20 keywords used by author.
        """
        if self.text == None:
            print("Must get author text before Rx list (use method: findText)")
            return
        return keywords(self.text)
    
    def getGender(self):
        """
        Must first get self.flair with getInfo method;
        Gets gender - Male or Female - when provided;
        Not currently able to identify transgendered individuals
        """
        if self.flair == None:
            print("""Must get author info before sex, M/F info 
                  (use method: getInfo)""")
            return
        elif self.flair == "No author info available":
            self.gen = "-"
            return self.gen
        gen_search = re.findall("""^(F|f|M|m)[^A-Za-z]|[^A-Za-z](F|f|M|m)[^A-Za-z]""",
                                self.flair)
        if gen_search:
            for item in gen_search[0]:
                if item:
                    self.gen = item.upper()
                    return self.gen
        else:
            self.gen = '-'
            return self.gen
    
    def getAgeDx(self):
        """
        Must first get self.flair with getInfo method
        Gets age and age / date of diagnosis, when provided. 
        This was designed to be most compatible with MS subreddit, and only 
        somewhat compatible with other communities at this time.
        (Would be ideal to apply a machine-learning method to use to 
        train the model to pick out relevant info from user variations
        which I hope to learn how to do in CLMS program!)
        """
        if self.flair == None:
            print("Must get author info before age/diagnosis info (use method: getInfo)")
            return
        elif self.flair == "No author info available":
            self.age = "-"
            self.dx = "-"
            return (self.age, self.dx)
        dx_search = re.findall('([dD][Xx].*[0-9]+[0-9]+|[Dd]iag.*[0-9]+[0-9]+|[Ss]ince.*[0-9]+)', 
                               self.flair)
        if dx_search:
            for item in dx_search:
                if item:
                    self.dx = item
        else:
            self.dx = '-'
        
        #Search for age after diagnosis info by removing diagnosis text 
        #from flair text; this eliminates some "false positives" by removing
        #numbers that could interfere with age search. This is the method
        #with the most errors in communities outside of Multiple Sclerosis
        #which use different Flair conventions, e.g. different vocabulary 
        #for indicating date of diagnosis
        try:
            age_search1 = self.flair.replace(self.dx,'')
        except:
            age_search1 = self.flair
        age_search = re.findall('^([1-9][0-9])[^0-9]|[^[0-9\']([1-9][0-9])[^0-9]', 
                                age_search1)
        if age_search:
            for item in age_search[0]:
                if item:
                    self.age = item
        else:
            self.age = '-'
        return (self.age, self.dx)
    
    def getRx(self):
        """
        Must first get self.flair with getInfo method;
        Gets person's current treatment(s) when provided 
        """
        if self.flair == None:
            print("Must get author info before treatment (use method: getInfo)")
            return
        elif self.flair == "No author info available":
            self.rx = '-'
            return self.rx
        
        self.rx = ''
        flair_tokens = wordsClean(self.flair)
        rxDict = redditRx(flair_tokens)
        if rxDict:
            for key in rxDict:
                self.rx = self.rx + key + ' '
        else:
            self.rx = '-'
        return self.rx
    
    def __repr__(self):
        return self.author
    
    def __str__(self):
        return "Reddit username: " + self.author

###########################

#Here begins the part of the code that produces the analysis that is
#printed out in the console. It first asks the user to choose one of four
#disease states to analyze:

diseaseState = input("""Welcome! 
This program will analyze recent Reddit
conversations in a community (subreddit) 
for one of the medical conditions below.
                     
Choose disease state to analyze: 
1 - Multiple Sclerosis
2 - Diabetes
3 - Psoriasis 
4 - Crohn's Disease
>> """)
    
try:
    if int(diseaseState) == 1:
        subreddit = reddit.subreddit('MultipleSclerosis')
        treatmentDict = MSTreatmentDict
    elif int(diseaseState) == 2:
        subreddit = reddit.subreddit('Diabetes')
        treatmentDict = DiaTreatmentDict
    elif int(diseaseState) == 3:
        subreddit = reddit.subreddit('Psoriasis')
        treatmentDict = PsTreatmentDict
    elif int(diseaseState) == 4:
        subreddit = reddit.subreddit('CrohnsDisease')
        treatmentDict = CDtreatmentDict
except:
    subreddit = reddit.subreddit('MultipleSclerosis')
    treatmentDict = MSTreatmentDict
    print("""Your response wasn't recognized. 'Multiple Sclerosis' will
    be analyzed""")

#Next asks user for number of Reddit threads to analyze, with a limit
#of 20. In reality, users would probably be most interested in analyzing
#posts for specific time period, but for now this is based on number of posts
n = input("""How many threads would you like to import? (Limit of 20) 
>>  """)

try:
    n = int(n)
    if n > 20: 
        n = 20
        print("""The maximum number of threads (20) will be analyzed""")
except: 
    n = 20
    print("""The maximum number of threads (20) will be analyzed""")

#call functions to get necessary lists and dictionaries for output
j = threads(n)
jtxt = redditText(j)
jwords = set(jtxt)
jwordsunc = uncommonWords(jwords)
jrx = redditRx(jwordsunc)
jauthors = authorCount(j)

#Analysis printed out below

print("")
print("""Table of Contents\n1. Number of posts & comments\n2. Dates of posts
3. Treatment Counts\n4. Treatment variations and misspellings
5. Top 20 keywords\n6. Top Authors""")

print("")
print("- 1 - ")
print("")
thrdCount = [int(i[0]) for i in j]
maxx = max(thrdCount)
print("Total threads analyzed: %d" % maxx)
num = len(j)
print("Number of Reddit posts & comments analyzed: %d" % num)

#prints plot of dates
print("")
print("- 2 -")
datesPlot(j)

#prints list of treatments mentioned, accounting for common misspellings
print("")
print("- 3 -")
treatmentCount(jtxt, jrx)

print("")
print("- 4 -")
print("")
print("TREATMENT VARIATIONS (& MISSPELLINGS):")
for k,v in jrx.items():
    print(k, end=": ")
    for rx in v:
        print(rx, end = " ")
    print("")

#Top 20 keywords for the whole reddit corpus
print("")
print("- 5 -")
print("")
jkeyw = keywords(jtxt)
print("TOP 20 KEYWORDS       WORD COUNT")
for pos, kv in enumerate(jkeyw):
    try:
        spaces = (20 - len(kv[0]))*' '
    except:
        spaces = '  '
    print(str(pos+1)+'.',kv[0],spaces,kv[1])

#Here is a basic table showing the top 20 authors with information about them, 
#including number of posts, age, gender, date of diagnosis, and top keyword 
#they use (plus count of that keyword)
print("")
print("- 6 -")
print("")
print("TOP 20 AUTHORS")
print("")
tab = tt.Texttable()
headings = ['Author','Posts','Age','Gender','Dx Date','Current Rx','Top Keyword [count]']
tab.header(headings)
authors = topAuthors(jauthors)
posts = []
age = []
gender = []
dx = []
rx = []
kws = []
for author in authors:
    author = redditor(author)
    author.getInfo(j)
    author.getText(j)
    posts.append(author.getPosts(j))
    a,d = author.getAgeDx()
    age.append(a)
    dx.append(d)
    gender.append(author.getGender())
    rx.append(author.getRx())
    kw = author.getKeywords()
    kw = kw[0]
    kw = kw[0]+" ["+str(kw[1])+"]"
    kws.append(kw)

for row in zip(authors,posts,age,gender,dx,rx,kws):
    tab.add_row(row)

author_table = tab.draw()
print (author_table)
