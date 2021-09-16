# TensorFlow.js 예제: 웹사이트 URL이 피싱인지 정상인지 분류하기

이 예제는 [피싱 웹사이트 데이터셋](http://eprints.hud.ac.uk/id/eprint/24330/6/MohammadPhishing14July2015.pdf)을 사용해
URL이 [피싱](https://ko.wikipedia.org/wiki/%ED%94%BC%EC%8B%B1)인지 아닌지 분류하는 방법을 보여줍니다.
주어진 데이터셋의 샘플을 피싱인지 아닌지 두 그룹으로 분류하기 때문에 이 문제는 이진 분류입니다.

## 특성

이 데이터셋에는 다음과 같은 특성이 있습니다:

1. `HAVING_IP_ADDRESS`: 도메인 이름 대신 IP 주소가 사용되는지 여부. {-1, 1}
2. `URL_LENGTH`: URL 길이가 적정한지, 의심스러운지, 피싱인지. {-1, 0, 1}
3. `SHORTINING_SERVICE`: URL 단축 서비스를 사용하는지 여부. {-1, 1}
4. `HAVING_AT_SYMBOL`: URL이 "@" 기호를 포함하는지 여부. {-1, 1}
5. `DOUBLE_SLASH_REDIRECTING`: URL이 이중 슬래시 리다이렉팅을 포함하는지 여부. {-1, 1}
6. `PREFIX_SUFFIX`: URL이 "-"로 구분되는 접두사 혹은 접미사를 포함하는지 여부. {-1, 1}
7. `HAVING_SUB_DOMAIN`: URL에 있는 서브 도메인의 개수가 적정한지, 의심스러운지, 피싱인지. {-1, 0, 1}
8. `SSLFINAL_STATE`: URL이 https를 사용하고 인증기관을 신뢰할 수 있는지, https를 사용하지만 신뢰할 수 없는 인증기관인지, https를 사용하지 않는지. {-1, 0, 1}
9. `DOMAIN_REGISTERATION_LENGTH`: 도메인이 1년 이내에 만료되는지 여부. {-1, 1}
10. `FAVICON`: favicon이 외부 도메인에서 로드되는지 여부. {-1, 1}
11. `PORT`: 표준 포트를 사용하지 않는지 여부. {-1, 1}
12. `HTTPS_TOKEN`: `https` 토큰이 도메인의 일부인지 여부. {-1, 1}
13. `REQUEST_URL`: 외부 도메인에 대한 요청의 비율이 정상인지 의심스러운지 여부. {-1, 1}
14. `URL_OF_ANCHOR`: 외부 도메인이나 자신의 도메인을 참조하는 앵커 태그에 있는 URL 비율이 정상인지, 의심스러운지, 피싱인지. {-1, 0, 1}
15. `LINKS_IN_TAGS`: 외부 도메인을 참조하는 메타, 스크립트, 링크 태그의 비율이 정상인지, 의심스러운지, 피싱인지. {-1, 0, 1}
16. `SFH`: 서버 폼 핸들러(server form handler)가 비어있거나 "about: blank"을 담고 있거나, 다른 도메인을 참조하거나, 정상인지. {1, 0, -1}
17. `SUBMITTING_TO_EMAIL`: 폼이 정보를 이메일로 전송하는지 여부. {-1, 1}
18. `ABNORMAL_URL`: URL이 호스트 이름을 포함하고 있지 않는지 여부. {-1, 1}
19. `REDIRECT`: URL이 1개 페이지 이내로 리다이렉팅되는지, 2와 4페이지 사이인지, 4페이지보다 많은지. {-1, 0, 1}
20. `ON_MOUSEOVER`: `onMouseOver`가 상태 막대를 바꾸는지 여부. {-1, 1}
21. `RIGHTCLICK`: 오른쪽 클릭이 막혀있는지 여부. {-1, 1}
22. `POPUPWIDNOW`: 파법 윈도우가 텍스트 필드를 포함하는지 여부. {-1, 1}
23. `IFRAME`: 페이지가 iframe 태그를 포함하는지 여부. {-1, 1}
24. `AGE_OF_DOMAIN`: 도메인 이력이 6개월 미만인지 여부. {-1, 1}
25. `DNSRECORD`: 도메인에 대한 DNS 레코드가 있는지 여부. {-1, 1}
26. `WEB_TRAFFIC`: 웹사이트 랭킹이 100,000 미만인지, 100,000 보다 큰지, Alexa에 기록되지 않거나 웹 트래픽이 없는지. {-1, 0, 1}
27. `PAGE_RANK`: 페이지 랭크 0.2보다 작은지 여부. {-1, 1}
28. `GOOGLE_INDEX`: 웹 페이지 구글에 인덱스 되어 있지 않은지. {-1, 1}
29. `LINKS_POINTING_TO_PAGE`: 페이지를 가리키는 링크가 2개 보다 큰지, 0~2개 사이인지, 0개인지. {-1, 0, 1}
30. `STATISTICAL_REPORT`: 호스트가 상위 피싱 IP나 도메인에 속하는지 여부. {-1, 1}

## 실행 방법

```sh
$ npx http-server
```
