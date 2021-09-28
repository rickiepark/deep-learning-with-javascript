let recognizer;

function predictWord() {
 // 음성 인식기를 훈련한 단어 배열.
 const words = recognizer.wordLabels();
 recognizer.listen(({scores}) => {
   // 점수를 (점수, 단어) 쌍의 리스트로 변환합니다.
   scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
   // 가장 높은 확률의 단어를 찾습니다.
   scores.sort((s1, s2) => s2.score - s1.score);
   document.querySelector('#console').textContent = scores[0].word;
 }, {probabilityThreshold: 0.75});
}

async function app() {
 recognizer = speechCommands.create('BROWSER_FFT');
 await recognizer.ensureModelLoaded();
 console.log(recognizer.wordLabels())
 document.querySelector('#note').textContent = '모델이 준비되었습니다. 다음 단어 중 하나를 말해 보세요. '+recognizer.wordLabels().slice(2)
 predictWord();
}

app();
