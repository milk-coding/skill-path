const express = require('express');
const path = require('path');
const app = express();

// 현재 디렉토리에서 정적 파일 서빙
app.use(express.static(__dirname));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index02.html'));
});

app.listen(8080, function(){ //포트번호 8080으로 들어왔을 때, 서버를 띄우라는 의미
    console.log('listening on 8080')//javaScript는 세미콜론 제거거
}); //서버 어디다가 열 지를 정해줌.







/* //누군가가 /pet 으로 방문하면, 
//pet 관련 안내문을 띄워주자.

app.get('/pet', function(request, response){
    response.send('펫용품 쇼핑할 수 있는 페이지입니다.');

}); // get 요청을 수행하는 기계완성.

app.get('/', function(request, response){
    response.sendFile(__dirname + '/index02.html')

}); //이게 하나만 쓰면 홈이 됨.
 */