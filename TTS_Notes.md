### Notes from TTS Experimentation

For the TTS Pipeline, *all* of the top models from HuggingFace were tried. (https://huggingface.co/spaces/TTS-AGI/TTS-Arena)

The goal was to use the models that were easy to setup and sounded less robotic with ability to include sound effects like laughter, etc.


#### Kokoro
```
# 1️⃣ Install kokoro
!pip install -q kokoro>=0.8.2 soundfile
# 2️⃣ Install espeak, used for English OOD fallback and some non-English languages
!apt-get -qq -y install espeak-ng > /dev/null 2>&1
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇧🇷 'p' => Brazilian Portuguese pt-br

# 3️⃣ Initalize a pipeline
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice

# This text is for demonstration purposes only, unseen during training
text = '''
The sky above the port was the color of television, tuned to a dead channel.
"It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."
It was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.

These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come. One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures. The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire. Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need. We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.

[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''
# text = '「もしおれがただ偶然、そしてこうしようというつもりでなくここに立っているのなら、ちょっとばかり絶望するところだな」と、そんなことが彼の頭に思い浮かんだ。'
# text = '中國人民不信邪也不怕邪，不惹事也不怕事，任何外國不要指望我們會拿自己的核心利益做交易，不要指望我們會吞下損害我國主權、安全、發展利益的苦果！'
# text = 'Los partidos políticos tradicionales compiten con los populismos y los movimientos asamblearios.'
# text = 'Le dromadaire resplendissant déambulait tranquillement dans les méandres en mastiquant de petites feuilles vernissées.'
# text = 'ट्रांसपोर्टरों की हड़ताल लगातार पांचवें दिन जारी, दिसंबर से इलेक्ट्रॉनिक टोल कलेक्शनल सिस्टम'
# text = "Allora cominciava l'insonnia, o un dormiveglia peggiore dell'insonnia, che talvolta assumeva i caratteri dell'incubo."
# text = 'Elabora relatórios de acompanhamento cronológico para as diferentes unidades do Departamento que propõem contratos.'

# 4️⃣ Generate, display, and save audio files in a loop.
generator = pipeline(
    text, voice='af_heart', # <= change voice here
    speed=1, split_pattern=r'\n+'
)
for i, (gs, ps, audio) in enumerate(generator):
    print(i)  # i => index
    print(gs) # gs => graphemes/text
    print(ps) # ps => phonemes
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000) # save each audio file
```
Under the hood, kokoro uses misaki, a G2P library at https://github.com/hexgrad/misaki
Model Facts
Architecture:

StyleTTS 2: https://arxiv.org/abs/2306.07691
ISTFTNet: https://arxiv.org/abs/2203.02395
Decoder only: no diffusion, no encoder release

#### Parler-TTS

Minimal code to run their models:

```
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# Define text and description
text_prompt = "This is where the actual words to be spoken go"
description = """
Laura's voice is expressive and dramatic in delivery, speaking at a fast pace with a very close recording that almost has no background noise.
"""

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

ipd.Audio(audio_arr, rate=model.config.sampling_rate)
```

The really cool aspect of these models are the ability to prompt the `description` which can change the speaker profile and pacing of the outputs.

Surprisingly, Parler's mini model sounded more natural.

In their [repo](https://github.com/huggingface/parler-tts/blob/main/INFERENCE.md#speaker-consistency) they share names of speakers that we can use in prompt.

#### Suno/Bark

Minimal code to run bark:

```
voice_preset = "v2/en_speaker_6"
sampling_rate = 24000

text_prompt = """
Exactly! [sigh] And the distillation part is where you take a LARGE-model,and compress-it down into a smaller, more efficient model that can run on devices with limited resources.
"""
inputs = processor(text_prompt, voice_preset=voice_preset).to(device)

speech_output = model.generate(**inputs, temperature = 0.9, semantic_temperature = 0.8)
Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)
```

Similar to parler models, suno has a [library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) of speakers.

v9 from their library sounded robotic so we use Parler for our first speaker and the best one from bark.

The incredible thing about Bark model is being able to add sound effects: `[Laugh]`, `[Gasps]`, `[Sigh]`, `[clears throat]`, making words capital causes the model to emphasize them. 

Adding `-` gives a break in the text. We utilize this knowledge when we re-write the transcript using the 8B model to add effects to our transcript.

Note: Authors suggest using `...`. However, this didn't work as effectively as adding a hyphen during trails.

#### Hyper-parameters: 

Bark models have two parameters we can tweak: `temperature` and `semantic_temperature`

Below are the notes from a sweep, prompt and speaker were fixed and this was a vibe test to see which gives best results. `temperature` and `semantic_temperature` respectively below:

First, fix `temperature` and sweep `semantic_temperature`
- `0.7`, `0.2`: Quite bland and boring
- `0.7`, `0.3`: An improvement over the previous one
- `0.7`, `0.4`: Further improvement 
- `0.7`, `0.5`: This one didn't work
- `0.7`, `0.6`: So-So, didn't stand out
- `0.7`, `0.7`: The best so far
- `0.7`, `0.8`: Further improvement 
- `0.7`, `0.9`: Mix feelings on this one

Now sweeping the `temperature`
- `0.1`, `0.9`: Very Robotic
- `0.2`, `0.9`: Less Robotic but not convincing
- `0.3`, `0.9`: Slight improvement still not fun
- `0.4`, `0.9`: Still has a robotic tinge
- `0.5`, `0.9`: The laugh was weird on this one but the voice modulates so much it feels speaker is changing
- `0.6`, `0.9`: Most consistent voice but has a robotic after-taste
- `0.7`, `0.9`: Very robotic and laugh was weird
- `0.8`, `0.9`: Completely ignore the laughter but it was more natural
- `0.9`, `0.9`: We have a winner probably

After this about ~30 more sweeps were done with the promising combinations:

Best results are at ```speech_output = model.generate(**inputs, temperature = 0.9, semantic_temperature = 0.8)
Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)```


### Notes from other models that were tested:

Promising directions to explore in future:

- [MeloTTS](https://huggingface.co/myshell-ai/MeloTTS-English) This is most popular (ever) on HuggingFace
- [WhisperSpeech](https://huggingface.co/WhisperSpeech/WhisperSpeech) sounded quite natural as well
- [F5-TTS](https://github.com/SWivid/F5-TTS) was the latest release at this time, however, it felt a bit robotic
- E2-TTS: r/locallama claims this to be a little better, however, it didn't pass the vibe test
- [xTTS](https://coqui.ai/blog/tts/open_xtts) It has great documentation and also seems promising

#### Some more models that weren't tested:

In other words, we leave this as an exercise to readers :D

- [Fish-Speech](https://huggingface.co/fishaudio/fish-speech-1.4)
- [MMS-TTS-Eng](https://huggingface.co/facebook/mms-tts-eng)
- [Metavoice](https://huggingface.co/metavoiceio/metavoice-1B-v0.1)
- [Hifigan](https://huggingface.co/nvidia/tts_hifigan)
- [TTS-Tacotron2](https://huggingface.co/speechbrain/tts-tacotron2-ljspeech) 
- [MMS-TTS-Eng](https://huggingface.co/facebook/mms-tts-eng)
- [VALL-E X](https://github.com/Plachtaa/VALL-E-X)
