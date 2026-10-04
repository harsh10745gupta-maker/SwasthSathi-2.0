
document.addEventListener("DOMContentLoaded",()=>{
  const form=document.querySelector("#assessment-form");
  const loader=document.querySelector("#loader");
  if(form && loader){
    form.addEventListener("submit",()=>{
      loader.classList.add("show");
    });
  }
  document.querySelectorAll("[data-count]").forEach(el=>{
    const target=parseInt(el.dataset.count||"0",10); let n=0;
    const step=Math.max(1,Math.ceil(target/35));
    const timer=setInterval(()=>{n+=step;if(n>=target){n=target;clearInterval(timer)}el.textContent=n},30);
  });
});
