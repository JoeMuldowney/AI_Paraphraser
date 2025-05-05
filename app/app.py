import os
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'app_model', 'epoch3_lrate2e_b48_s15000v3000')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

max_input = 1024

graphs_app = Flask(__name__)
graphs_app.secret_key = 'graph_flask_app'

# Sample summaries for testing/demo
sample_metrics = [
    {
        "Comet": "0.77",
        "Bert": "0.879",
        'Rouge1': 0.396,
        'rouge2': 0.115,
        'rougeL': 0.188,

        "input": """When is bug spray more than just bug spray? When it's a compound that, according to researchers at Vanderbilt University, is thousands of times stronger than DEET, works on many different insects and could very well save lives. Scientists at the school say they've developed just such a repellent. Known merely as VUAA1 for now, they say it works not just on mosquitoes, but also ants, flies, moths and a host of other bugs that, at best, are a nuisance and, at worst, carry deadly diseases like malaria. In fact, the project began as an effort to curb malaria, which will probably be contracted by as many as 500 million people this year, said Laurence Zwiebel, a professor in the biological science department at Vanderbilt University in Nashville. But researchers soon discovered the chemical compound they'd created could have wider uses. "It turns out if we found the world's greatest mosquito repellent, no one would care," Zweibel said. "So we needed to find something that would work against all insects." The trick, Zweibel says, was taking the way bug spray as we know it works, then doing the exact opposite. Most commercial insect repellents target the bug's olfactory system the way it smells to make it harder for them to find us and consider us as targets for a meal. "We decided to take a more aggressive approach and, rather than turn off the mosquito's olfactory system, we could look for something that would turn it too far on, to see if we could design a new generation of insect repellents based on overloading their smell system," Zweibel said. "They hate, just like we hate, overstimulation. They will move away from too much smell." The researchers, who were funded in part by a grant from the Bill & Melinda Gates Foundation, say that, so far, VUAA1 has worked on every insect they've tested it on. That widens the possibilities for it. Potential uses range from shooing away bugs that eat crops to a commercial product that can be used to keep pests out of the home. But the original goal, helping prevent a deadly disease that, according to the Centers for Disease Control killed more than 660,000 people in 2010, remains foremost for Zwiebel and his team. "Our hope is that we're able to help develop a product that can be sold for profit in the developed world, and use that profit to leverage distribution in the developing world," he said. "Our hope is that every time we spray on a mosquito repellent here in America, we're subsidizing malaria reduction in Africa and Asia." The team has patented the compound. But there's no word yet on when it might hit the shelves. For now, they're doing tests to make sure it will be safe for people to use. "So far, we don't see any toxic effects," Zwiebel said.""",
        "expected": """An insect repellent from Vanderbilt works on mosquitoes, ants and other bugs. They say it's thousands of times more powerful than DEET. It works by flooding the bug's system of smell. They've filed for a patent, but testing needed before it hits the market.""",
        "output": """Researchers at Vanderbilt University say they've developed just such a repellent. Known merely as VUAA1 for now, they say it works not just on mosquitoes, but also ants, flies, moths and a host of other bugs. Potential uses range from shooing away bugs that eat crops to a commercial product that can keep pests out of the home."""
},
    {
        "Comet": "0.800",
        "Bert": "0.895",
        "Rouge1": 0.325,
        'rouge2': 0.142,
        'rougeL': 0.232,
        "input": """Russia's ambassador to Britain was summoned to the UK's Foreign Office on Thursday to explain why two Russian bombers were flying over the English Channel this week, a representative of the office said. Two Russian bombers "caused disruption to civil aviation" when they flew near, but did not cross into, British airspace on Wednesday, the Foreign Office representative said. Two UK Royal Air Force jets intercepted the bombers, both capable of carrying nuclear weapons, south of Bournemouth, England, over the English Channel, a UK defense spokesman said Thursday. The British jets escorted the bombers for about an hour-and-a-half until the bombers left the area, the defense spokesman said. "Russian aircraft maneuvers yesterday are part of  increasing pattern of out-of-area operations by Russian aircraft," the Foreign Office representative said. The representative did not elaborate on how civil aviation was disrupted. Details about what the Russian ambassador has told Britain about the incident weren't immediately available. Wednesday's incident would be the latest in what NATO has said is an increase in Russian military flights near alliance members' territory. In November, NATO said its members' jets had been scrambled more than 400 times in 2014 to intercept Russian military flights close to members' territories a 50% increase over the previous year. The increase harkens back to the days of the Cold War, NATO Secretary General Jens Stoltenberg said November 20 during a visit to NATO member Estonia. "This pattern is risky and unjustified," Stoltenberg said. "So NATO remains vigilant. We are here. And we are ready to defend all allies against any threat." In a November report, the European Leadership Network listed more than 40 "close military encounters between Russia and the West" in the eight months from March to October. Three of those, including a near-collision between a Russian military plane and a Swedish passenger aircraft carrying 132 people, were classified as "high-risk" incidents that could have led to direct military confrontation between Russia and the West, according to the report, titled "Dangerous Brinksmanship." UK jets intercepted more than 100 Russian aircraft last year, according to the British defense ministry. CNN's Lindsay Isaac and Brad Lendon contributed to this report.""""",
        "expected": """Russian bombers disrupted civil aviation, UK Foreign Office representative says. UK: Russian bombers were flying over English Channel. Russian aircraft didn't cross into British airspace, spokesman says.""",
        "output": """Two Russian bombers "caused disruption to civil aviation" when they flew near, but did not cross into, British airspace. Two UK Royal Air Force jets intercepted the bombers south of Bournemouth, England, over the English Channel. The incident would be the latest in what NATO has said is an increase in Russian military flights near alliance members' territory."""
    },
    {
        "Comet": "0.908 (Highest observed from sample of testing data)",
        "Bert": "0.976 (Highest observed from sample of testing data)",
        "Rouge1": 0.845,
        'rouge2': 0.753,
        'rougeL': 0.845,
        "input": """Facebook has launched a page devoted to U.S. politics, located at Facebook.com/USpolitics. Rolled out at the Personal Democracy Forum in New York City, the page monitors how U.S. politicians, elected officials and political campaigns use Facebook to connect with citizens. Facebook is an important part of U.S. politics today. Barack Obama has an incredibly popular page, with more than 8.6 million fans. Recently, U.S. President George W. Bush started his own Facebook page, amassing over 70,000 fans in a very short timespan. Facebook also has a similar page devoted to sports, as well as all of the celebrities who use Facebook. © 2010 MASHABLE.com. All rights reserved.""",
        "expected": """Facebook has launched a page devoted to U.S. politics. The page monitors how politicians use Facebook to connect with citizens. Barack Obama has more than 8.6 million fans.""",
        "output": """Facebook has launched a page devoted to U.S. politics. The page monitors how politicians, elected officials and political campaigns use Facebook to connect with citizens. President Barack Obama has an incredibly popular page, with more than 8.6 million fans.""",
    },
    {
        "Comet": "0.818",
        "Bert": "0.944",
        "Rouge1": 0.677,
        'rouge2': 0.606,
        'rougeL': 0.661,
        "input": """A hit man who authorities said was hired by a woman whose three husbands have all died under unusual circumstances has been convicted of second-degree murder in her second spouse's shooting. Alfred 'Terry' Everette, of Mississippi, was convicted on Thursday of gunning down Ernest Smith, 38, in 2006, at the behest of Smith's wife, 50-year-old Emma Raine. Raine has pleaded not guilty in the case and is awaiting trial in March on a second-degree murder charge. Hit man Alfred 'Terry' Everette , of Mississippi, was convicted of killing Ernest Smith, 38, in 2006, at the behest of Smith's wife, 50-year-old Emma Raine  Mr Smith, a New Orleans preacher, was the second of Raine's three husbands. She has been charged only in Smith's death, but a prosecutor made it clear that the other two deaths were suspicious. 'Her main source of income was killing husbands,' said Assistant District Attorney Laura Rodrigue, who prosecuted the case with her father, District Attorney Leon Cannizzaro. Emma Raine's third husband, James Raine, 37, was shot in their Pearl River, Mississippi, home in 2011. News reports at the time said police were investigating the shooting as a homicide, and that Emma Raine had been out of town at the time. Her first husband, Leroy Evans, was run over by a car in Vicksburg, Mississippi, and then choked to death in his sleep about a year later, in 1994, after his feeding tube was removed. Complicating the case James Raine has been implicated posthumously in Ernest Smith's death. Prosecutors said James Raine was Emma's boyfriend while she was married to Smith. He also was the foster brother of hit man Everette. Authorities says James Raine had promised Everette up to $100,000 from an $800,000 expected life insurance payment to kill Smith, who was shot outside his New Orleans home. During the trial, a foster brother, Enoch Raine, and two uncles, William and Henry Fowler, all testified that Everette admitted killing Smith when they confronted him two days after James Raine was fatally shot. Everette's defense attorneys, Michael Kennedy and Tanzanika Ruffin, attacked the credibility of Raine's relatives, saying they hoped to benefit from his life insurance policy. 'If James' policy isn't paid to Emma, who gets the money? Ruffin asked the jurors. 'The family does, don't they? Of course they do.' Outside the courtroom, the relatives denied Ruffin's allegations. 'It was never about money,' said Enoch Raine, a Mississippi firefighter. 'Emma and Terry need to be brought to justice for this,' said William Fowler. 'It has messed with our family. We wanted justice for this. We also wanted justice for James.'""",
        "expected": """Alfred 'Terry' Everette was convicted of killing Ernest Smith, 38, in 2006 at the behest of Smith's wife, 50-year-old Emma Raine. She has pleaded not guilty in the case and is awaiting trial in March on a second-degree murder charge. Mr Smith, a New Orleans preacher, was the second of Raine's three husbands. She has been charged only in Smith's death, but a prosecutor made it clear that the other two deaths were suspicious.""",
        "output": """Alfred 'Terry' Everette, of Mississippi, was convicted of killing Ernest Smith, 38, at the behest of Smith's wife, 50-year-old Emma Raine. He has pleaded not guilty in the case and is awaiting trial in March on a second-degree murder charge.""",
    },
    {
        "Comet": "0.845 ",
        "Bert": "0.926",
        "Rouge1": 0.553,
        'rouge2': 0.391,
        'rougeL': 0.468,
        "input": """You know that friend who has an opinion or a joke about everything? Sure, they're entertaining, but sometimes you wish you could get them to shut up. Now on Twitter, you can. Twitter is rolling out a "mute" feature that will let you silence certain users in your feed. Once you've muted them, their tweets and retweets will no longer be visible in your timeline, and you won't receive their push or SMS notifications, although @ replies and mentions will still appear. "In the same way you can turn on device notifications so you never miss a tweet from your favorite users, you can now mute users you'd like to hear from less," Twitter said in a blog post late Monday. The mute function will be available on Twitter's iPhone and Android apps and its website and will roll out to all Twitter users "in the coming weeks." Twitter says muted users will still be able to favorite, reply to and retweet your tweets, but you won't see any of that activity in your timeline. Muted users you follow can still send you a direct message. But the muted user won't know you've muzzled them, and you can unmute them at any time. In this way, the mute tool is similar to Facebook's feature that lets you hide friends' updates without unfriending them and, presumably, hurting their feelings. The change is Twitter's latest attempt to give its 200 million-plus users greater control over their interactions on the social platform. The company must hope it will be better received than its last similar update. In December, Twitter tweaked its settings that allow users to block others who harass or annoy them. The change allowed the blocked user to still see the profile and tweets of the person who blocked them and to retweet their posts. But after a furious user backlash, Twitter abruptly reversed itself. Unlike muted users, blocked users can't follow you on Twitter. Some Twitter users were questioning the need for a mute feature Tuesday, saying people should just unfollow users who annoy them. "I find the #twittermute thing hilarious. It's just the Internet switch it off and go for a walk!" wrote Stewart Lee, a Web officer for a foundation in the United Kingdom. Others, of course, just made jokes. "Forget Klout, now we need a 'Muzzle' ranking showing your Follower-Mute ratio," said Christian Christensen, a university professor in Stockholm. To mute a user, click "more" next to their tweet and then "mute @username." To mute someone from their profile page, click the gear icon on the page and choose "mute @username.""",
        "expected": """Twitter is rolling out a "mute" feature that will let you silence certain users. The muted user won't know you've muzzled them, and you can unmute them anytime. New function will roll out to all Twitter users "in the coming weeks""",
        "output": """Twitter is rolling out a "mute" feature that will let you silence certain users in your feed. Once you've muted them, their tweets and retweets will no longer be visible in your timeline. Muted users can still send you a direct message, but you can unmute them at any time.""",
    },
    {
        "Comet": "0.567",
        "Bert": "0.880",
        "Rouge1": 0.329,
        'rouge2': 0.233,
        'rougeL': 0.303,
        "input": """Two American hikers held in Iran and accused of spying after straying across the unmarked border there have been married in California, an attorney said. Sarah Shourd and Shane Bauer married Saturday near the ocean, surrounded by some 200 family and friends, Ben Rosenfeld, a San Francisco attorney, said in a statement. "Friends and family assumed roles at their wedding, in the same spirit of community which suffuses Sarah and Shane's everyday lives, and which characterized the tireless broad support they received throughout their ordeal. Friends traveled from around the globe to be present. The group danced outside under the amorous cross-rays of Venus and a 'supermoon,'" he said. The newlyweds have since left for their honeymoon. Shourd, Bauer and a third hiker Josh Fattal were arrested after straying across the unmarked border between Iraqi Kurdistan and Iran in July 2009. Bauer proposed during their time in prison, fashioning an engagement ring from a thread of one of his shirts. Shourd was released in 2010, and Bauer and Fattal were freed last year after 781 days in captivity and a trial for espionage that Bauer said was based on 'ridiculous lies.'""",
        "expected": """Sarah Shourd and Shane Bauer married Saturday near the ocean. They, along with Josh Fattal, were arrested after straying across the border in 2009. Bauer proposed during their time in prison. Shourd was released in 2010, and Bauer and Fattal were freed last year.""",
        "output": """Sarah Shourd and Shane Bauer married Saturday near the ocean. Friends and family assumed roles at their wedding. Friends traveled from around the globe to be present. The newlyweds have since left for their honeymoon.""",
    },
    {
        "Comet": "0.512",
        "Bert": "0.838",
        "Rouge1": 0.325,
        'rouge2': 0.142,
        'rougeL': 0.232,
        "input": """William Holden, Alec Guinness, George C. Scott, Denzel Washington: They all won Academy Awards for roles as heroic military men. But when James Maitland Stewart enlisted in the U.S. Army in 1941, he already had a Best Actor Oscar to his name, for his indelible performance as reporter Mike Connor in "The Philadelphia Story." Based in England during much of his service, Jimmy Stewart flew dozens of combat missions more than a few as command pilot on sorties deep into Nazi-occupied Europe and returned from the war with honors that included the Distinguished Flying Cross and Distinguished Service Medal. In September 1945, LIFE photographer Peter Stackpole accompanied Stewart around his Indiana, Pennsylvania, hometown, chronicling the goings-on when the movie star returned home a hero. Read the full story and see the entire gallery on Life.com.""",
        "expected": """Jimmy Stewart put his career on hold to join the Army Air Corps. The actor was well into his 30s and had already been rejected for being too skinny. He was one of few Americans ever to rise from private to colonel in four years.""",
        "output": """James Maitland Stewart enlisted in the U.S. Army in 1941. He flew dozens of combat missions more than a few as command pilot on sorties deep into Nazi-occupied Europe. In September 1945, LIFE photographer Peter Stackpole accompanied Stewart around his Indiana, Pennsylvania, hometown.""",
    },
    {
        "Comet": "0.377 (Lowest observed from sample of testing data)",
        "Bert": "0.820 (Lowest observed from sample of testing data)",
        "Rouge1": 0.027,
        'rouge2': 0.0,
        'rougeL': 0.027,
        "input": """You probably never met Harry Stamps. I never met him either. And, sadly, now he's gone. Harry left this world last Saturday. He was 80. Of course, there's no real reason why any of us should know Harry. He's just some guy from Long Beach, Mississippi. Though I say that with complete reverence. "Just some guy" is usually the one who helps you fix the lawn mower. Or looks after your dog. Or loans you his truck so you can go to Costco and buy 80 cases of pudding and maybe some lobster dip. I have needs. A truck would be helpful. But while most of the world never got to meet the man, now, thanks to the Internet, countless thousands know his name. And it's all because of one of the greatest obituaries ever written. When Amanda Lewis sat down to eulogize her father there was no way she'd know her words would go viral. Generally speaking, obituaries don't get wildly passed around online, for they tend to lack cats. Which Harry hated. "He wouldn't know what going viral means,"Amanda told the local Sun Herald newspaper. "He would have thought that was a disease he contracted, which would have excited him to have another illness to lord over folks." After all, Harry never lost in "competitive sickness." So, Amanda's humorous and touching obit for her dad quickly spread throughout Facebook and other social media sites, and readers were treated to wonderful sentiments about the man she described as a foodie and a natty dresser. To this first point that of him being a foodie Amanda notes his membership in the Bacon of the Month Club. The fact that this even exists should give us all hope. On the other hand, Harry also loved a martini glass filled with buttermilk, garnished with a chunk of cornbread, which does seem rather weird.  Weird or not, Amanda added that, "As a point of pride, he purported to remember every meal he had eaten in his 80 years of life." And when it came to style, there wasn't a runway in the world that would have him. Unless it was, say, an actual runway. "His signature everyday look was all his: A plain pocketed T-shirt designed by the fashion house Fruit of the Loom ... his black-label elastic waist shorts worn above the navel and sold exclusively at the Sam's on Highway 49." But most importantly, Harry loved his grass-stained Mississippi State University baseball cap. It was kind of his thing. And, somehow, when you look back on a man's life, that just sounds way better than, "Yeah, Dad was really into Brooks Brothers." Not that there's anything wrong with Brooks Brothers there isn't but it's sort of funny how, in the end, something like that grass-stained MSU baseball cap seems to matter. So long as it's not too gross. "Ol' Wilbur wore that very same undershirt for 60 straight years. Never washed it. You could smell him from Tucson." Yes, just like that hat, Harry enjoyed the simple things in life, and he had incredible wanderlust for the natural world around him. Amanda wrote that her dad traveled extensively. "He only stayed in the finest quality AAA-rated campgrounds," the obit reads. "Many years later he purchased a used pop-up camper for his family to travel in style, which spoiled his daughters for life." But there were things that Harry didn't like. Besides cats, he also couldn't stand "Law & Order," Martha Stewart, and daylight saving time. To the latter, Amanda made sure to point out, "It is not lost on his family that he died the very day that he would have had to spring his clock forward. This can only be viewed as his final protest." The whole obituary is filled with little gems like these, and there's nothing I can write here that will do it any justice. You simply need to read it for yourself. It's brilliant. When I emailed Amanda asking for photos of her dad, she wrote me back and said, "This whole thing is surreal. My dad was the most authentic person I have ever known. It tickles me that this Every Man has resonated with so many people." And at the obituary's conclusion, Amanda hoped her Every Man father could resonate, perhaps, a little more. It ends with a plea. "Finally, the family asks that in honor of Harry that you write your Congressman and ask for the repeal of Daylight Saving Time. Harry wanted everyone to get back on the Lord's time." Of course, Washington politicians probably won't listen to any protests in Harry's name. He was just some guy from Long Beach, Mississippi..
                """,
        "expected": """Harry Stamps passed away on Saturday, March 9. His daughter wrote an obituary that went viral. Obituary: Harry never lost in "competitive sickness." Harry hated daylight saving time.""",
        "output": """"Just some guy" is usually the one who helps you fix the lawn mower. Or loans you his truck so you can go to Costco and buy 80 cases of pudding and maybe some lobster dip. But thanks to the Internet, countless thousands know his name.""",
},
    {
        "Comet": "0.786",
        "Bert": "0.892",
        "Rouge1": 0.361,
        'rouge2': 0.2,
        'rougeL': 0.277,
        "input": """Sudanese President Omar al-Bashir says he will accept the results of a referendum this month that could see the country split in two. Al-Bashir said "peace is our ultimate goal" with southern Sudan, which could become an independent state after the January 9 vote. He called on the government in the south to provide a safe environment for the referendum. "The referendum process shall go on with God's blessings, with the trust of our commitment that we will renew at this moment," al-Bashir said, "and accepting the result that will come from the desire of the citizens and their choices." Al-Bashir made the comments in a speech Friday marking Sudan's 55th Independence Day. He also promised to negotiate what comes after the referendum. "Our acceptance of the final results will not be withdrawn or hesitated about," he said, "because the peace is our ultimate goal in our relationships with our southern brothers, even if they choose a path other than unity." The referendum is part of a 2005 peace agreement that ended two decades of violence between the north and oil-rich south. The conflict led to the deaths of 2 million people, many from starvation. The impending vote has sparked fears of renewed violence. Al-Bashir has been leader of Sudan since 1989. He is wanted by the International Criminal Court in The Hague, Netherlands, on allegations of war crimes and genocide in western Sudan's Darfur region, where violence that erupted in 2003 has left at least 300,000 people dead.""",
        "expected": """"Peace is our ultimate goal," President Omar al-Bashir says. He will abide by referendum result, he says during speech. Southern Sudan will vote this month on whether to become independent state.""",
        "output": """Sudanese President Omar al-Bashir says he will accept the results of a referendum this month. The vote could see the country split in two. The referendum is part of a 2005 peace agreement that ended two decades of violence.""",
    },
]

@graphs_app.route("/", methods=['GET', 'POST'])
def home():
    index = int(request.args.get('index', 0))
    total_samples = len(sample_metrics)
    sample = sample_metrics[index % total_samples]

    summary = ""

    if request.method == 'POST':
        model_input = request.form['input_text']
        total_tokens = get_count(model_input)
        if total_tokens > 1024:
            summary = "Input larger than expected. Please try again with a smaller input (under 1024 tokens)."
        else:
            summary = summarize(model_input)

    else:
        # On GET request with index, show the sample summary in the textarea
        model_input = sample['output']

    return render_template(
        "home.html",
        input_text=model_input,
        summary=summary,
        sample_metrics=sample,
        index=index,
        total=total_samples
    )
def get_count(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

def summarize(summary):
    # Consistent preprocessing
    inputs = tokenizer(
        summary,
        truncation=True,
        padding="max_length",
        max_length=max_input,
        return_tensors='pt'
    )

    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=150,
        min_length=20,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode and return summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

if __name__ == "__main__":
    graphs_app.run(host='0.0.0.0', port=5000)
