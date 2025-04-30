function openModal(modalId) {
  const modal = document.getElementById(modalId);
  const modalContent = modal.querySelector(".modal-content");
  modal.classList.remove("hidden");
  setTimeout(() => {
    modal.classList.add("opacity-100");
    modalContent.classList.remove("scale-95", "opacity-0");
    modalContent.classList.add("scale-100", "opacity-100");
  }, 10);
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  const modalContent = modal.querySelector(".modal-content");
  modalContent.classList.remove("scale-100", "opacity-100");
  modalContent.classList.add("scale-95", "opacity-0");
  modal.classList.remove("opacity-100");
  setTimeout(() => {
    modal.classList.add("hidden");
  }, 300);
}
