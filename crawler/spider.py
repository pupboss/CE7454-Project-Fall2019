from lxml import etree
import requests
import pandas as pd
import os
from pyquery import PyQuery as pq

tt = pd.read_csv('../data/movie_2000_to_2018.tsv', usecols=[0], delimiter='\t')
tt = tt.values.tolist()

for step, each_tt in enumerate(tt):

	print(each_tt, step)

	movie_url = 'https://www.imdb.com/title/' + each_tt[0]
	imdb_html = pq(url = movie_url, encoding='utf-8')

	Budget = imdb_html('h4:contains("Budget:")')
	if Budget:
		Budget = Budget[0].tail.strip(' ').split('\n')[0]
	else:
		continue
		
	Box_Office = imdb_html('h4:contains("Cumulative Worldwide Gross")')
	if Box_Office:
		Box_Office = Box_Office[0].tail.strip(' ')
	else:
		continue

	Story_Line = imdb_html('div[id=titleStoryLine]>div[class="inline canwrap"]').text().strip('\t').strip('\n')
	if not Story_Line:
		continue

	Stars = imdb_html('h4:contains("Stars:")').siblings()
	if len(Stars) == 5:
		Star_1 = Stars[0].text
		Star_1_nm = Stars[0].attrib['href'].split('/')[2]
		Star_2 = Stars[1].text
		Star_2_nm = Stars[1].attrib['href'].split('/')[2]
		Star_3 = Stars[2].text
		Star_3_nm = Stars[2].attrib['href'].split('/')[2]
	else:
		continue

	image_url = imdb_html('div[class="poster"]>a')
	if not image_url:
		continue
	image_url = 'https://www.imdb.com' + image_url[0].attrib['href']
	rm_num = image_url.split('/')[-1]
	poster = pq(url=image_url, encoding='utf-8')
	print(rm_num)
	poster_url = poster('script')[1].text.split(rm_num)[1].split('"src":"')[1].split(".jpg")[0] + '.jpg'
	print(poster_url)
	r = requests.get(poster_url)

	if not os.path.isdir("./posters"):
		os.makedirs('./posters')

	with open("./posters/" + each_tt[0] + ".jpg",'wb') as f:
		f.write(r.content)

	with open('./spider_info.tsv','a') as f:
		row = (each_tt[0], Budget, Box_Office, Star_1, Star_2, Star_3, Star_1_nm, Star_2_nm, Star_3_nm)
		row = '\t'.join(row)
		f.write(row+'\n')

	with open('./spider_story.tsv','a') as f:
		row = (each_tt[0], Story_Line)
		row = '\t'.join(row)
		f.write(row+'\n')


















