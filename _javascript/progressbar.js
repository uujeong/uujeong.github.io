// 스크롤 이벤트 리스너를 추가합니다.
window.onscroll = function () {
  updateProgressBar();
};

function updateProgressBar() {
  // 문서의 전체 높이와 뷰포트의 높이를 계산합니다.
  var docHeight = document.documentElement.scrollHeight;
  var viewHeight = window.innerHeight;

  // 사용자가 스크롤한 양을 계산합니다.
  var scrolled = window.scrollY;

  // 사용자가 스크롤할 수 있는 최대 높이를 계산합니다.
  var scrollableHeight = docHeight - viewHeight;

  // 프로그레스 바의 너비를 계산합니다. (스크롤한 양 / 스크롤할 수 있는 최대 높이) * 100
  var progressBarWidth = (scrolled / scrollableHeight) * 100;

  // 프로그레스 바의 너비를 업데이트합니다.
  document.getElementById('indicator').style.width = progressBarWidth + '%';
}
