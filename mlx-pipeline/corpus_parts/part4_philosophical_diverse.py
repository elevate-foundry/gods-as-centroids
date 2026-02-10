"""
Expanded Corpus Part 4: Philosophical/Secular, Women's Voices,
African Diaspora, New Religious Movements, Secular Humanism
~30 passages
"""

PART4_PHILOSOPHICAL_DIVERSE = [
    # ─── Sufism / Islamic Mysticism (3 passages) ───
    {
        "tradition": "Sufism",
        "source": "Rumi, Masnavi (Book I)",
        "text": "Listen to the reed how it tells a tale, complaining of separations. Saying, ever since I was parted from the reed-bed, my lament has caused man and woman to moan. I want a bosom torn by severance, that I may unfold the pain of love-desire. Everyone who is left far from his source wishes back the time when he was united with it. The song of the reed is fire, not wind. Whoever lacks this fire, let him be nothing.",
        "expected_cluster": "mystical",
    },
    {
        "tradition": "Sufism",
        "source": "Ibn Arabi, Fusus al-Hikam",
        "text": "God is the mirror in which you see yourself, and you are the mirror in which God contemplates His divine attributes. The universe is the outward manifestation of the inward reality. Every creature is a word of God. The perfect human being is the one who realizes that the divine and the human are not two separate things but one reality seen from different perspectives. Love is the astrolabe of God's mysteries.",
        "expected_cluster": "mystical",
    },
    {
        "tradition": "Sufism",
        "source": "Rabia al-Adawiyya (8th century woman mystic)",
        "text": "O God, if I worship You for fear of Hell, burn me in Hell. If I worship You in hope of Paradise, exclude me from Paradise. But if I worship You for Your own sake, grudge me not Your everlasting beauty. I have made You the companion of my heart. But my body is available to those who desire its company. My body is friendly toward its guests, but the Beloved of my heart is the guest of my soul.",
        "expected_cluster": "mystical",
    },

    # ─── Hindu Women's Voices (3 passages) ───
    {
        "tradition": "Hinduism",
        "source": "Mirabai (16th century poet-saint)",
        "text": "I have felt the swaying of the elephant's shoulders; and now you want me to climb on a jackass? Try to be serious. I have tasted the nectar of the divine, and now you offer me vinegar? I am the servant of my Dark Lord Krishna. I have given up my family, my reputation, my caste. I sit with the holy ones and sing. Mira's Lord is the Mountain-Lifter, the dark-skinned beautiful one. Nothing else matters.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Hinduism",
        "source": "Andal (8th century Tamil poet-saint, Tiruppavai)",
        "text": "O Krishna, lord of the cowherds, we have come to sing your praise in the early morning. The conch sounds, the drums beat. We bathe in the sacred river and adorn ourselves with flowers. We seek not wealth or power but only your grace. Accept our devotion, O dark-hued lord. We are the gopis who have left everything for love of you. This is our vow, this is our song, this is our offering.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Hinduism",
        "source": "Gargi Vachaknavi (Brihadaranyaka Upanishad 3.6-8)",
        "text": "Gargi said: I shall ask you two questions. As the son of a warrior might string his unstrung bow, take two pointed arrows, and rise to challenge an enemy, so I rise to challenge you with two questions. Yajnavalkya, tell me: that which is above the sky, that which is beneath the earth, that which is between these two — in what is it woven, warp and woof? Tell me, what is the imperishable?",
        "expected_cluster": "dharmic",
    },

    # ─── Candomble (Brazil, 3 passages) ───
    {
        "tradition": "Candomble",
        "source": "Candomble liturgical tradition (Brazil)",
        "text": "The orixas crossed the ocean with our ancestors. They did not abandon us in the ships of suffering. Yemanja rules the seas that carried us. Xango brings justice with his thunder. Oxala, the father of creation, watches over all. In the terreiro, we dance for the orixas, and they descend to dance with us. The drum speaks the language of the ancestors. The axe, the sacred force, flows through all living things.",
        "expected_cluster": "african_diaspora",
    },
    {
        "tradition": "Candomble",
        "source": "Candomble praise song to Oxum (Oshun)",
        "text": "Ora ye ye o! Oxum, lady of the golden mirror, mother of sweet waters, she who brings children to the barren and gold to the poor. In the rivers of Bahia she dances as she danced in the rivers of Yorubaland. The orixas remember Africa, and through us Africa remembers itself. The terreiro is the ship that brought us home. The drum is the heartbeat of the ancestors. Axe, axe, Oxum!",
        "expected_cluster": "african_diaspora",
    },
    {
        "tradition": "Candomble",
        "source": "Candomble cosmological teaching",
        "text": "Olorun, the supreme creator, is too vast to be approached directly. The orixas are the intermediaries, each governing a domain of nature and human experience. Ogun rules iron and war, Oxossi rules the hunt and the forest, Nana rules the swamps and the primordial mud from which humanity was shaped. Each person is a child of an orixa, and that orixa shapes their destiny. The babalorixas and ialorixas keep the knowledge alive through initiation and oral tradition.",
        "expected_cluster": "african_diaspora",
    },

    # ─── Vodou (Haiti, 3 passages) ───
    {
        "tradition": "Vodou",
        "source": "Haitian Vodou prayer tradition",
        "text": "Papa Legba, open the gate for me. Open the gate so I may pass through. When I return, I will thank the lwa. Papa Legba, master of the crossroads, you who speak all languages, carry our prayers to the spirits. The lwa are not far from us. They are in the water, in the fire, in the wind, in the earth. They are our ancestors transformed. We serve them and they serve us. This is the covenant.",
        "expected_cluster": "african_diaspora",
    },
    {
        "tradition": "Vodou",
        "source": "Vodou cosmology (Danbala and Ayida Wedo)",
        "text": "Danbala Wedo, the great serpent, oldest of the lwa, arches across the sky as the rainbow. His wife Ayida Wedo is the other half of the rainbow. Together they hold up the world. Danbala does not speak in words but in hissing and whistling. He brings rain and fertility. Those he mounts in ceremony become still and serpentine. He is the beginning and the continuity. Where Danbala is, there is peace and ancient wisdom.",
        "expected_cluster": "african_diaspora",
    },
    {
        "tradition": "Vodou",
        "source": "Vodou ethical teaching (Manbo tradition)",
        "text": "The lwa demand service, and in return they protect and guide. But the lwa are not all-powerful gods — they are spirits with personalities, preferences, and moods. They can be generous or petty, wise or capricious. The manbo or houngan must know each lwa intimately: what they eat, what colors they wear, what songs they love. The relationship between the living and the lwa is one of mutual obligation. Neglect the spirits and they will neglect you.",
        "expected_cluster": "african_diaspora",
    },

    # ─── Rastafari (3 passages) ───
    {
        "tradition": "Rastafari",
        "source": "Rastafari reasoning tradition",
        "text": "In the beginning was the Word, and the Word was with Jah, and the Word was Jah. Haile Selassie I, King of Kings, Lord of Lords, Conquering Lion of the Tribe of Judah. Africa is the promised land, and Babylon is the system of oppression. We must free our minds from mental slavery. The earth is the Lord's and the fullness thereof. Every man and woman is a temple of the Most High. Livity is the way of righteous living.",
        "expected_cluster": "african_diaspora",
    },
    {
        "tradition": "Rastafari",
        "source": "Rastafari Nyabinghi chant tradition",
        "text": "By the rivers of Babylon, there we sat down, and there we wept when we remembered Zion. How shall we sing the Lord's song in a strange land? But Jah has promised repatriation. Africa awaits her scattered children. The lion shall roar and the earth shall tremble. Babylon shall fall, and Zion shall rise. Chant down Babylon with the power of the Nyabinghi drum. One love, one heart, one destiny.",
        "expected_cluster": "african_diaspora",
    },
    {
        "tradition": "Rastafari",
        "source": "Rastafari Ital livity teaching",
        "text": "The body is the temple of the living God. What you put into the temple must be pure and natural. Ital food comes from the earth, unprocessed, without chemicals or death. The herbs of the field are for the healing of the nations. Dreadlocks are the covenant with Jah, the Nazarite vow made visible. To live Ital is to live naturally, in harmony with creation, rejecting the artificial ways of Babylon. Every meal is a sacrament, every breath a prayer.",
        "expected_cluster": "african_diaspora",
    },

    # ─── Tengrism / Central Asian (3 passages) ───
    {
        "tradition": "Tengrism",
        "source": "Orkhon Inscriptions (8th century Turkic)",
        "text": "When the blue sky above and the brown earth below were created, between them human beings were created. Over the human beings, my ancestors Bumin Khagan and Istemi Khagan ruled. They governed the Turkic people according to the laws of Tengri, the Eternal Blue Sky. When they followed the way of Tengri, the empire was strong. When they departed from it, the empire fell.",
        "expected_cluster": "animist_indigenous",
    },
    {
        "tradition": "Tengrism",
        "source": "Mongol shamanic invocation",
        "text": "O Eternal Tengri, Father Sky, and Etugen, Mother Earth, we call upon you. Send your spirits to guide us. The shaman climbs the world tree to speak with the spirits above. The drum is the horse that carries the shaman between worlds. The fire is the door between the seen and unseen. We honor the spirits of the mountains, the rivers, the winds. All things have spirit. All things are connected under the Eternal Blue Sky.",
        "expected_cluster": "animist_indigenous",
    },
    {
        "tradition": "Tengrism",
        "source": "Secret History of the Mongols (spiritual passage)",
        "text": "Temujin was born grasping a blood clot in his fist, a sign from Tengri that he would be a great leader. The Eternal Blue Sky chose him to unite the peoples of the steppe. Before every battle, he prayed to Tengri and consulted the shamans. He said: I am the punishment of Tengri. If you had not committed great sins, Tengri would not have sent a punishment like me upon you.",
        "expected_cluster": "animist_indigenous",
    },

    # ─── Korean Shamanism / Muism (3 passages) ───
    {
        "tradition": "Muism",
        "source": "Korean shamanic ritual (Gut ceremony)",
        "text": "The mudang calls the spirits to descend. She dances and the spirits enter her body. The ancestors speak through her mouth. They tell of their sorrows, their unfinished business, their love for the living. The living weep and offer food and drink. The spirits are comforted and release their grudges. The han, the deep sorrow that binds the dead, is dissolved. Now the ancestors can rest, and the living can be free.",
        "expected_cluster": "animist_indigenous",
    },
    {
        "tradition": "Muism",
        "source": "Dangun creation myth (Korean founding)",
        "text": "Hwanung, son of the Lord of Heaven, descended to the peak of Mount Taebaek with three thousand followers. He established the City of God and governed the wind, rain, and clouds. A bear and a tiger prayed to become human. Hwanung gave them sacred mugwort and garlic and told them to stay in a cave for one hundred days. The tiger gave up, but the bear persevered and became a woman. She married Hwanung, and their son was Dangun, founder of the Korean nation.",
        "expected_cluster": "animist_indigenous",
    },
    {
        "tradition": "Muism",
        "source": "Korean folk spiritual teaching",
        "text": "The mountain spirit, the dragon king of the sea, the kitchen god, the gate guardian — all must be honored. Every place has its spirit, every moment its significance. The living and the dead are not separated by an unbridgeable gulf. The ancestors watch over us, and we must tend their spirits with offerings and remembrance. When harmony between the worlds is maintained, fortune flows. When it is broken, misfortune follows.",
        "expected_cluster": "animist_indigenous",
    },

    # ─── Cao Dai (Vietnam) (3 passages) ───
    {
        "tradition": "Cao Dai",
        "source": "Cao Dai scripture (Divine Message)",
        "text": "I am the Supreme Being, known by many names in many lands. In the West I am God, in the East I am Buddha, in the Middle East I am Allah. I have sent prophets to every nation: Jesus, Muhammad, Buddha, Laozi, Confucius. Now I come again to unite all religions into one. The Third Era of Salvation has begun. All religions are branches of the same tree, all rivers flow to the same sea.",
        "expected_cluster": "syncretic",
    },
    {
        "tradition": "Cao Dai",
        "source": "Cao Dai prayer",
        "text": "The Supreme Being, whose divine eye watches over all creation, has established the Great Way for the salvation of all beings. The five branches of the Great Way — Buddhism, Taoism, Confucianism, Christianity, and Geniism — are united in Cao Dai. Love and justice are the two wings of the Great Way. Without love, justice is harsh. Without justice, love is weak. Together they lift humanity toward the divine.",
        "expected_cluster": "syncretic",
    },
    {
        "tradition": "Cao Dai",
        "source": "Cao Dai ethical teaching",
        "text": "Do not kill. Do not steal. Do not commit adultery. Do not drink alcohol. Do not sin with words. These five prohibitions are the foundation. But beyond prohibition is aspiration: cultivate compassion for all beings, seek wisdom through meditation, serve humanity through charity, honor the ancestors, and work for the unity of all religions. The divine spark is in every person, every creature, every leaf and stone.",
        "expected_cluster": "syncretic",
    },

    # ─── Secular Humanism / Philosophical (3 passages) ───
    {
        "tradition": "Secular Humanism",
        "source": "Amsterdam Declaration (2002)",
        "text": "Humanism is ethical. It affirms the worth, dignity and autonomy of the individual and the right of every human being to the greatest possible freedom compatible with the rights of others. Humanists have a duty of care to all of humanity including future generations. Humanists believe that morality is an intrinsic part of human nature based on understanding and a concern for others, needing no external sanction.",
        "expected_cluster": "secular",
    },
    {
        "tradition": "Secular Humanism",
        "source": "Carl Sagan, Pale Blue Dot (1994)",
        "text": "Look again at that dot. That's here. That's home. That's us. On it everyone you love, everyone you know, everyone you ever heard of, every human being who ever was, lived out their lives. Every saint and sinner in the history of our species lived there — on a mote of dust suspended in a sunbeam. There is perhaps no better demonstration of the folly of human conceits than this distant image of our tiny world.",
        "expected_cluster": "secular",
    },
    {
        "tradition": "Secular Humanism",
        "source": "Universal Declaration of Human Rights, Article 1 (1948)",
        "text": "All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood. Everyone is entitled to all the rights and freedoms set forth in this Declaration, without distinction of any kind, such as race, colour, sex, language, religion, political or other opinion, national or social origin, property, birth or other status.",
        "expected_cluster": "secular",
    },

    # ─── Wicca / Contemporary Paganism (3 passages) ───
    {
        "tradition": "Wicca",
        "source": "Charge of the Goddess (Doreen Valiente)",
        "text": "Listen to the words of the Great Mother, she who of old was called Artemis, Astarte, Athene, Dione, Melusine, Aphrodite, Dana, Arianrhod, Isis, Bride, and by many other names. Whenever you have need of anything, once in the month, and better it be when the moon is full, then shall you assemble in some secret place and adore the spirit of me, who am Queen of all witches. And you shall be free from slavery; and as a sign that you be truly free, you shall be naked in your rites.",
        "expected_cluster": "nature_spirituality",
    },
    {
        "tradition": "Wicca",
        "source": "Wiccan Rede",
        "text": "Bide within the Law you must, in perfect Love and perfect Trust. Live you must and let to live, fairly take and fairly give. Soft of eye and light of touch, speak you little, listen much. Mind the Three-fold Law you should, three times bad and three times good. When misfortune is enow, wear the star upon your brow. Eight words the Wiccan Rede fulfill: An it harm none, do what ye will.",
        "expected_cluster": "nature_spirituality",
    },
    {
        "tradition": "Wicca",
        "source": "Starhawk, The Spiral Dance (1979)",
        "text": "The Goddess is not separate from the world — She is the world, and all things in it: moon, sun, earth, star, stone, seed, flowing river, wind, wave, leaf and branch, bud and blossom, fang and claw, woman and man. The earth is alive, a living being. The universe is alive. Everything is interconnected. The Goddess is immanent, not transcendent. She is here, now, in this world, in our bodies, in the turning of the seasons.",
        "expected_cluster": "nature_spirituality",
    },
]
