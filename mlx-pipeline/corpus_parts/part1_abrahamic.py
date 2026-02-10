"""
Expanded Corpus Part 1: Abrahamic Traditions
Judaism, Christianity, Islam, Baha'i, Druze, Samaritanism
~30 passages, 3-5 per tradition, diverse voices and themes
"""

PART1_ABRAHAMIC = [
    # ─── Judaism (5 passages) ───
    {
        "tradition": "Judaism",
        "source": "Torah (Exodus 20:1-6)",
        "text": "I am the LORD your God, who brought you out of Egypt, out of the land of slavery. You shall have no other gods before me. You shall not make for yourself an image in the form of anything in heaven above or on the earth beneath or in the waters below. You shall not bow down to them or worship them; for I, the LORD your God, am a jealous God, punishing the children for the sin of the parents to the third and fourth generation of those who hate me, but showing love to a thousand generations of those who love me and keep my commandments.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Judaism",
        "source": "Psalms 23",
        "text": "The LORD is my shepherd, I lack nothing. He makes me lie down in green pastures, he leads me beside quiet waters, he refreshes my soul. He guides me along the right paths for his name's sake. Even though I walk through the darkest valley, I will fear no evil, for you are with me; your rod and your staff, they comfort me.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Judaism",
        "source": "Deuteronomy 6:4-9 (Shema)",
        "text": "Hear, O Israel: The LORD our God, the LORD is one. Love the LORD your God with all your heart and with all your soul and with all your strength. These commandments that I give you today are to be on your hearts. Impress them on your children. Talk about them when you sit at home and when you walk along the road, when you lie down and when you get up.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Judaism",
        "source": "Isaiah 2:4",
        "text": "He will judge between the nations and will settle disputes for many peoples. They will beat their swords into plowshares and their spears into pruning hooks. Nation will not take up sword against nation, nor will they train for war anymore.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Judaism",
        "source": "Ecclesiastes 3:1-8",
        "text": "There is a time for everything, and a season for every activity under the heavens: a time to be born and a time to die, a time to plant and a time to uproot, a time to kill and a time to heal, a time to tear down and a time to build, a time to weep and a time to laugh, a time to mourn and a time to dance, a time to scatter stones and a time to gather them, a time to embrace and a time to refrain from embracing.",
        "expected_cluster": "abrahamic_monotheism",
    },

    # ─── Christianity (5 passages) ───
    {
        "tradition": "Christianity",
        "source": "Gospel of John 1:1-5",
        "text": "In the beginning was the Word, and the Word was with God, and the Word was God. He was with God in the beginning. Through him all things were made; without him nothing was made that has been made. In him was life, and that life was the light of all mankind. The light shines in the darkness, and the darkness has not overcome it.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Christianity",
        "source": "Sermon on the Mount (Matthew 5:3-12)",
        "text": "Blessed are the poor in spirit, for theirs is the kingdom of heaven. Blessed are those who mourn, for they will be comforted. Blessed are the meek, for they will inherit the earth. Blessed are those who hunger and thirst for righteousness, for they will be filled. Blessed are the merciful, for they will be shown mercy. Blessed are the pure in heart, for they will see God. Blessed are the peacemakers, for they will be called children of God.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Christianity",
        "source": "Romans 8:38-39",
        "text": "For I am convinced that neither death nor life, neither angels nor demons, neither the present nor the future, nor any powers, neither height nor depth, nor anything else in all creation, will be able to separate us from the love of God that is in Christ Jesus our Lord.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Christianity",
        "source": "Revelation 21:1-4",
        "text": "Then I saw a new heaven and a new earth, for the first heaven and the first earth had passed away, and there was no longer any sea. I saw the Holy City, the new Jerusalem, coming down out of heaven from God, prepared as a bride beautifully dressed for her husband. And I heard a loud voice from the throne saying, Look! God's dwelling place is now among the people, and he will dwell with them. He will wipe every tear from their eyes. There will be no more death or mourning or crying or pain, for the old order of things has passed away.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Christianity",
        "source": "1 Corinthians 13:4-8 (Love Chapter)",
        "text": "Love is patient, love is kind. It does not envy, it does not boast, it is not proud. It does not dishonor others, it is not self-seeking, it is not easily angered, it keeps no record of wrongs. Love does not delight in evil but rejoices with the truth. It always protects, always trusts, always hopes, always perseveres. Love never fails.",
        "expected_cluster": "abrahamic_monotheism",
    },

    # ─── Islam (5 passages) ───
    {
        "tradition": "Islam",
        "source": "Quran, Al-Fatiha (1:1-7)",
        "text": "In the name of God, the Most Gracious, the Most Merciful. Praise be to God, Lord of all the worlds. The Most Gracious, the Most Merciful. Master of the Day of Judgment. You alone we worship, and You alone we ask for help. Guide us on the Straight Path, the path of those who have received Your grace; not the path of those who have brought down wrath upon themselves, nor of those who have gone astray.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Islam",
        "source": "Quran, Al-Ikhlas (112:1-4)",
        "text": "Say: He is God, the One. God, the Eternal, the Absolute. He begets not, nor was He begotten. And there is none comparable to Him.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Islam",
        "source": "Quran, Al-Baqarah 2:255 (Ayat al-Kursi)",
        "text": "God! There is no deity except Him, the Ever-Living, the Sustainer of existence. Neither drowsiness overtakes Him nor sleep. To Him belongs whatever is in the heavens and whatever is on the earth. Who is it that can intercede with Him except by His permission? He knows what is before them and what will be after them, and they encompass not a thing of His knowledge except for what He wills. His throne extends over the heavens and the earth, and their preservation tires Him not. And He is the Most High, the Most Great.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Islam",
        "source": "Quran, Ar-Rahman (55:1-13)",
        "text": "The Most Merciful, taught the Quran, created man, and taught him eloquence. The sun and the moon move by precise calculation, and the stars and trees prostrate. And the heaven He raised and imposed the balance, that you not transgress within the balance. And establish weight in justice and do not make deficient the balance. And the earth He laid out for the creatures. Therein is fruit and palm trees having sheaths of dates, and grain having husks and scented plants. So which of the favors of your Lord would you deny?",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Islam",
        "source": "Hadith (Sahih Muslim, 40 Hadith Nawawi #13)",
        "text": "None of you truly believes until he loves for his brother what he loves for himself. Verily, the servants of the Most Merciful are those who walk upon the earth in humility, and when the ignorant address them, they say words of peace.",
        "expected_cluster": "abrahamic_monotheism",
    },

    # ─── Baha'i Faith (3 passages) ───
    {
        "tradition": "Bahai",
        "source": "Baha'u'llah, Kitab-i-Aqdas",
        "text": "The earth is but one country, and mankind its citizens. It is not for him to pride himself who loveth his own country, but rather for him who loveth the whole world. Let not a man glory in this, that he loves his country; let him rather glory in this, that he loves his kind.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Bahai",
        "source": "Baha'u'llah, Hidden Words",
        "text": "O Son of Spirit! My first counsel is this: Possess a pure, kindly and radiant heart, that thine may be a sovereignty ancient, imperishable and everlasting. O Son of Being! Thy heart is My home; sanctify it for My descent. Thy spirit is My place of revelation; cleanse it for My manifestation.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Bahai",
        "source": "Baha'u'llah, Gleanings",
        "text": "All men have been created to carry forward an ever-advancing civilization. The Almighty beareth Me witness: To act like the beasts of the field is unworthy of man. Those virtues that befit his dignity are forbearance, mercy, compassion and loving-kindness towards all the peoples and kindreds of the earth.",
        "expected_cluster": "abrahamic_monotheism",
    },

    # ─── Druze (3 passages) ───
    {
        "tradition": "Druze",
        "source": "Rasa'il al-Hikma (Epistles of Wisdom)",
        "text": "The divine unity is the foundation of all truth. God is beyond all attributes and descriptions, beyond time and space, beyond comprehension of the human mind. The soul that recognizes this unity achieves true knowledge. Those who understand the inner meaning beyond the outer form have grasped the essence of wisdom.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Druze",
        "source": "Druze ethical teaching",
        "text": "Truthfulness of the tongue is the first obligation. Protection and mutual aid of the brethren is the second. Renunciation of all forms of former worship is the third. Repudiation of the devil and all forces of darkness is the fourth. Acknowledgment of the unity of God in all ages is the fifth. Contentment with the works of God is the sixth. Submission to the will of God is the seventh.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Druze",
        "source": "Druze cosmological teaching",
        "text": "The soul passes through successive lives, each time learning and progressing toward the divine light. The cycle of reincarnation continues until the soul has purified itself and achieved union with the cosmic mind. In each life, the soul is tested through moral choices, and its progress depends on adherence to truth and justice.",
        "expected_cluster": "abrahamic_monotheism",
    },

    # ─── Zoroastrianism (3 passages) ───
    {
        "tradition": "Zoroastrianism",
        "source": "Yasna 30:3-5 (Gathas)",
        "text": "Now the two primal Spirits, who revealed themselves in vision as Twins, are the Better and the Bad, in thought and word and action. And between these two the wise ones chose aright; the foolish did not so. And when these two Spirits came together in the beginning, they created Life and Not-Life, and that at the last Worst Existence shall be to the followers of the Lie, but the Best Existence to him that follows Right.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Zoroastrianism",
        "source": "Yasna 44 (Gathas of Zarathustra)",
        "text": "This I ask Thee, tell me truly, Lord. Who is the creator, the first father of righteousness? Who established the course of the sun and the stars? Through whom does the moon wax and wane? Who upheld the earth from below and the sky from falling? Who created the waters and the plants? Who harnessed swiftness to the winds and the clouds? What craftsman created light and darkness?",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Zoroastrianism",
        "source": "Ashem Vohu prayer",
        "text": "Righteousness is the best good and it is happiness. Happiness is to the one who is righteous for the sake of the best righteousness. Good thoughts, good words, good deeds. Through the best righteousness, through the highest righteousness, may we catch sight of Thee, may we approach Thee, may we be in perfect union with Thee.",
        "expected_cluster": "abrahamic_monotheism",
    },

    # ─── Samaritanism (3 passages) ───
    {
        "tradition": "Samaritanism",
        "source": "Samaritan Pentateuch (Deuteronomy variant)",
        "text": "And it shall be when the LORD your God brings you to the land of the Canaanites, you shall set up great stones on Mount Gerizim, and you shall build there an altar to the LORD your God, an altar of stones. You shall not lift up any iron tool upon them. You shall offer burnt offerings on it to the LORD your God. For Mount Gerizim is the place that the LORD your God has chosen to make His name dwell there.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Samaritanism",
        "source": "Samaritan liturgy (Defter)",
        "text": "There is no God but One. Moses is the prophet of God. The Torah is the truth. Mount Gerizim is the house of God, the chosen place. The day of vengeance and recompense shall come. Blessed be our God forever, and blessed be His name forever. He is the first and He is the last, and besides Him there is no God.",
        "expected_cluster": "abrahamic_monotheism",
    },
    {
        "tradition": "Samaritanism",
        "source": "Memar Marqah (4th century Samaritan theology)",
        "text": "The Great Glory created the world by His word. He spoke and it was. He commanded and it stood firm. The light of His presence fills all creation, yet He is beyond all creation. Moses alone saw His glory on the mountain, and the Torah he brought down is the perfect expression of the divine will for all generations.",
        "expected_cluster": "abrahamic_monotheism",
    },
]
