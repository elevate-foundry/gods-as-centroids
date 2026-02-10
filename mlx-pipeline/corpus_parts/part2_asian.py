"""
Expanded Corpus Part 2: South & East Asian Traditions
Hinduism, Buddhism, Jainism, Sikhism, Daoism, Confucianism, Shinto
~35 passages, 3-5 per tradition
"""

PART2_ASIAN = [
    # ─── Hinduism (5 passages) ───
    {
        "tradition": "Hinduism",
        "source": "Bhagavad Gita 2:22-24",
        "text": "As a person puts on new garments, giving up old ones, the soul similarly accepts new material bodies, giving up the old and useless ones. The soul can never be cut to pieces by any weapon, nor burned by fire, nor moistened by water, nor withered by the wind. This individual soul is unbreakable and insoluble, and can be neither burned nor dried. It is everlasting, present everywhere, unchangeable, immovable and eternally the same.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Hinduism",
        "source": "Bhagavad Gita 11:32-33 (Vishvarupa)",
        "text": "I am time, the great destroyer of the worlds, and I have come here to destroy all people. With the exception of you, all the soldiers here on both sides will be slain. Therefore get up. Prepare to fight and win glory. Conquer your enemies and enjoy a flourishing kingdom. They are already put to death by My arrangement, and you can be but an instrument in the fight.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Hinduism",
        "source": "Isha Upanishad 1",
        "text": "The Lord is enshrined in the hearts of all. The Lord is the supreme Reality. Rejoice in him through renunciation. Covet nothing. All belongs to the Lord. Thus working may you live a hundred years. Thus alone will you work in real freedom.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Hinduism",
        "source": "Rig Veda 10.129 (Nasadiya Sukta, Hymn of Creation)",
        "text": "Then even nothingness was not, nor existence. There was no air then, nor the heavens beyond it. What covered it? Where was it? In whose keeping? Was there then cosmic water, in depths unfathomed? Then there was neither death nor immortality, nor was there then the torch of night and day. The One breathed windlessly and self-sustaining. There was that One then, and there was no other.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Hinduism",
        "source": "Devi Mahatmya 1.75-78 (Goddess tradition)",
        "text": "By you this universe is borne, by you this world is created. By you it is protected, O Devi, and you always consume it at the end. You are the supreme knowledge as well as the great delusion, the great intellect and contemplation, as also the great illusion. You are the great Goddess, the great Demoness. You are the primordial cause of everything. You are the three qualities. You are the dark night of periodic dissolution. You are the great night of final dissolution. And you are the terrible night of delusion.",
        "expected_cluster": "dharmic",
    },

    # ─── Buddhism (5 passages) ───
    {
        "tradition": "Buddhism",
        "source": "Dhammapada 1-5",
        "text": "All that we are is the result of what we have thought: it is founded on our thoughts, it is made up of our thoughts. If a man speaks or acts with an evil thought, pain follows him, as the wheel follows the foot of the ox that draws the carriage. If a man speaks or acts with a pure thought, happiness follows him, like a shadow that never leaves him.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Buddhism",
        "source": "Heart Sutra",
        "text": "Form is emptiness, emptiness is form. Emptiness is not separate from form, form is not separate from emptiness. Whatever is form is emptiness, whatever is emptiness is form. The same is true for feelings, perceptions, mental formations, and consciousness. All dharmas are marked with emptiness. They are neither produced nor destroyed, neither defiled nor immaculate, neither increasing nor decreasing.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Buddhism",
        "source": "Metta Sutta (Loving-Kindness)",
        "text": "May all beings be happy. May all beings be safe. May all beings be healthy. May all beings live with ease. Whatever living beings there may be, whether they are weak or strong, omitting none, the great or the mighty, medium, short or small, the seen and the unseen, those living near and far away, those born and to-be-born — may all beings be happy.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Buddhism",
        "source": "Lotus Sutra, Chapter 2 (Mahayana)",
        "text": "The Buddhas, the World-Honored Ones, appear in the world for one great reason alone: to cause living beings to open the door to the Buddha's wisdom and gain purity. They appear in the world to show living beings the Buddha's wisdom. They appear in the world to cause living beings to awaken to the Buddha's wisdom. They appear in the world to cause living beings to enter the path of the Buddha's wisdom.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Buddhism",
        "source": "Tibetan Book of the Dead (Bardo Thodol)",
        "text": "O nobly-born, when thy body and mind were separating, thou must have experienced a glimpse of the Pure Truth, subtle, sparkling, bright, dazzling, glorious, and radiantly awesome, in appearance like a mirage moving across a landscape in springtime. Do not be daunted thereby, nor terrified, nor awed. That is the radiance of thine own true nature. Recognize it.",
        "expected_cluster": "dharmic",
    },

    # ─── Jainism (3 passages) ───
    {
        "tradition": "Jainism",
        "source": "Tattvartha Sutra 1.1-4 (Umasvati)",
        "text": "Right faith, right knowledge, and right conduct together constitute the path to liberation. These are the three jewels. The soul is characterized by consciousness. The function of souls is to help one another. The cycle of birth, life, death, and rebirth is governed by karma. Every living being has a soul, and every soul is potentially divine.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Jainism",
        "source": "Acaranga Sutra 1.4.1 (Ahimsa teaching)",
        "text": "All breathing, existing, living, sentient creatures should not be slain, nor treated with violence, nor abused, nor tormented, nor driven away. This is the pure, unchangeable, eternal law which the clever ones who understand the world have proclaimed. Among the zealous and the non-zealous, among the faithful and the non-faithful, among the wise and the unwise, this is the highest law.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Jainism",
        "source": "Mahavira's teaching (Uttaradhyayana Sutra 28)",
        "text": "Do not injure, abuse, oppress, enslave, insult, torment, torture, or kill any creature or living being. In happiness and suffering, in joy and grief, regard all creatures as you regard your own self. The enlightened one who has conquered the passions and attained perfect knowledge sees all beings with equal eye. This is the essence of wisdom.",
        "expected_cluster": "dharmic",
    },

    # ─── Sikhism (4 passages) ───
    {
        "tradition": "Sikhism",
        "source": "Japji Sahib (Guru Nanak, Mul Mantar)",
        "text": "There is One God, whose name is Truth, the Creator, without fear, without hatred, timeless in form, beyond birth and death, self-existent, known by the Guru's grace. Meditate on the True Name. True in the beginning, true throughout the ages, true even now, and says Nanak, shall ever be true.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Sikhism",
        "source": "Guru Granth Sahib, Asa di Var",
        "text": "From the One Light, the entire universe welled up. So who is good, and who is bad? O people, O Siblings of Destiny, do not wander deluded by doubt. The Creator is in the creation, and the creation is in the Creator. He is fully filling all places. Recognize the Lord's Light within all, and do not consider social class or status; there are no classes or castes in the world hereafter.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Sikhism",
        "source": "Guru Granth Sahib (Guru Arjan Dev)",
        "text": "I do not keep fasts, nor do I observe the month of Ramadan. I serve only the One who will protect me in the end. The One Lord of the World is my God. He judges both Hindus and Muslims. I do not make pilgrimages to Mecca, nor do I worship at Hindu sacred shrines. I serve the One Lord, and not any other.",
        "expected_cluster": "dharmic",
    },
    {
        "tradition": "Sikhism",
        "source": "Sukhmani Sahib (Guru Arjan Dev)",
        "text": "God is in all hearts, and all hearts are in God. He is pervading everywhere and in all. He Himself is the Creator, and He Himself is the enjoyer. From Him all come, and into Him all merge again. He is the thread which holds all beings together. He is the ocean, and all are His waves.",
        "expected_cluster": "dharmic",
    },

    # ─── Daoism (4 passages) ───
    {
        "tradition": "Daoism",
        "source": "Tao Te Ching, Chapter 1",
        "text": "The Tao that can be told is not the eternal Tao. The name that can be named is not the eternal name. The nameless is the beginning of heaven and earth. The named is the mother of ten thousand things. Ever desireless, one can see the mystery. Ever desiring, one can see the manifestations. These two spring from the same source but differ in name; this appears as darkness. Darkness within darkness. The gate to all mystery.",
        "expected_cluster": "eastern_nontheistic",
    },
    {
        "tradition": "Daoism",
        "source": "Tao Te Ching, Chapter 76",
        "text": "A man is born gentle and weak. At his death he is hard and stiff. Green plants are tender and filled with sap. At their death they are withered and dry. Therefore the stiff and unbending is the disciple of death. The gentle and yielding is the disciple of life. Thus an army without flexibility never wins a battle. A tree that is unbending is easily broken. The hard and strong will fall. The soft and weak will overcome.",
        "expected_cluster": "eastern_nontheistic",
    },
    {
        "tradition": "Daoism",
        "source": "Zhuangzi, Chapter 2 (Discussion on Making All Things Equal)",
        "text": "The Way has never known boundaries; speech has no constancy. But because of the recognition of a this, there came to be boundaries. Let me tell you what the boundaries are. There is left, there is right, there are theories, there are debates. This is called the Eight Virtues. The sage embraces things. Ordinary men discriminate among them and parade their discriminations before others. So I say, those who discriminate fail to see.",
        "expected_cluster": "eastern_nontheistic",
    },
    {
        "tradition": "Daoism",
        "source": "Tao Te Ching, Chapter 25",
        "text": "There was something formless and perfect before the universe was born. It is serene. Empty. Solitary. Unchanging. Infinite. Eternally present. It is the mother of the universe. For lack of a better name, I call it the Tao. It flows through all things, inside and outside, and returns to the origin of all things. The Tao is great. The universe is great. Earth is great. Humanity is great. These are the four great powers.",
        "expected_cluster": "eastern_nontheistic",
    },

    # ─── Confucianism (3 passages) ───
    {
        "tradition": "Confucianism",
        "source": "Analerta 12.2 (The Golden Rule)",
        "text": "Zhonggong asked about humaneness. The Master said: When abroad, behave as if you were receiving a great guest. Employ the people as if you were assisting at a great sacrifice. Do not do to others what you would not have them do to you. Then there will be no resentment against you, whether it is the affairs of a state that you are handling or the affairs of a family.",
        "expected_cluster": "eastern_nontheistic",
    },
    {
        "tradition": "Confucianism",
        "source": "Doctrine of the Mean (Zhongyong), Chapter 1",
        "text": "What Heaven imparts to man is called human nature. To follow our nature is called the Way. Cultivating the Way is called education. The Way cannot be separated from us for a moment. What can be separated from us is not the Way. Therefore the superior man is cautious in the place where he is not seen, and apprehensive in the place where he is not heard.",
        "expected_cluster": "eastern_nontheistic",
    },
    {
        "tradition": "Confucianism",
        "source": "Great Learning (Daxue), Opening",
        "text": "The Way of the Great Learning lies in manifesting luminous virtue, in renewing the people, and in resting in the highest good. Those in antiquity who wished to manifest luminous virtue throughout the world first governed their states well. Wishing to govern their states well, they first regulated their families. Wishing to regulate their families, they first cultivated their persons. Wishing to cultivate their persons, they first rectified their hearts.",
        "expected_cluster": "eastern_nontheistic",
    },

    # ─── Shinto (3 passages) ───
    {
        "tradition": "Shinto",
        "source": "Kojiki (Record of Ancient Matters), Creation",
        "text": "At the beginning of heaven and earth, there came into existence in the Plain of High Heaven three deities: the deity Master-of-the-August-Center-of-Heaven, the High-August-Producing-Wondrous deity, and the Divine-Producing-Wondrous deity. These three deities were all born alone, and hid their persons. The land was young, resembling floating oil, and drifted like a jellyfish.",
        "expected_cluster": "eastern_nontheistic",
    },
    {
        "tradition": "Shinto",
        "source": "Norito (Shinto prayer for purification)",
        "text": "By the command of the sovereign ancestral gods and goddesses who divinely dwell in the Plain of High Heaven, the eight myriad deities were convoked in a divine convocation. They bestowed upon us the lands of the four quarters as peaceful lands, tranquil lands. The offenses and pollution shall be carried away and purified by the kami of the rapids, the kami of the ocean depths, the kami of the breathing-in.",
        "expected_cluster": "eastern_nontheistic",
    },
    {
        "tradition": "Shinto",
        "source": "Motoori Norinaga, Naobi no Mitama (Spirit of Rectification)",
        "text": "The kami are of many kinds. Some are noble and some are base, some are strong and some are weak, some are good and some are evil. The kami are not separate from this world. They dwell in the mountains, in the rivers, in the trees, in the rocks, in the wind and the rain. To be in harmony with the kami is to be in harmony with the natural world. Sincerity of heart is the way to know the kami.",
        "expected_cluster": "eastern_nontheistic",
    },
]
