document.addEventListener("DOMContentLoaded", () => {
  // LẤY PHẦN TỬ
  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const chatContainer = document.getElementById("chat-container");
  const sidebar = document.querySelector(".sidebar");
  const sidebarChat = document.getElementById("sidebar-chat");
  const content = document.getElementById("content");
  const toggleBtn = document.getElementById("btn");
  const moTooltip = document.querySelector(".mo-side-bar");
  const dongTooltip = document.querySelector(".dong-side-bar");

  // TRẠNG THÁI
  let isCollapsed = false;
  let currentSessionId = null;

  // ẨN TOOLTIP BAN ĐẦU
  moTooltip.style.opacity = "0";
  dongTooltip.style.opacity = "0";

  // QUẢN LÝ LỊCH SỬ
  const ChatHistory = {
    key: "chat_sessions",
    getAll() {
      return JSON.parse(localStorage.getItem(this.key) || "[]");
    },
    saveAll(data) {
      localStorage.setItem(this.key, JSON.stringify(data));
    },
    createSession(firstMsg) {
      const sessions = this.getAll();
      const id = Date.now().toString();
      const newSession = {
        id,
        title: truncateText(firstMsg),
        messages: [],
      };
      sessions.unshift(newSession);
      this.saveAll(sessions);
      this.renderToSidebar();
      return id;
    },
    addMessageToSession(sessionId, sender, text) {
      const sessions = this.getAll();
      const session = sessions.find((s) => s.id === sessionId);
      if (session) {
        session.messages.push({ sender, text });
        this.saveAll(sessions);
      }
    },
    renameSession(sessionId, newTitle) {
      const sessions = this.getAll();
      const session = sessions.find((s) => s.id === sessionId);
      if (session) {
        session.title = truncateText(newTitle);
        this.saveAll(sessions);
        this.renderToSidebar();
      }
    },
    deleteSession(sessionId) {
      let sessions = this.getAll();
      sessions = sessions.filter((s) => s.id !== sessionId);
      this.saveAll(sessions);
      this.renderToSidebar();
    },
    renderToSidebar() {
      sidebarChat.innerHTML = "";
      const sessions = this.getAll();
      sessions.forEach((s) => {
        const item = document.createElement("div");
        item.className = "sidebar-item";

        const title = document.createElement("span");
        title.className = "chat-title";
        title.textContent = truncateText(s.title);
        title.onclick = () => loadChatSession(s.id);

        // Nút 3 chấm menu
        const menu = document.createElement("i");
        menu.className = "fa-solid fa-ellipsis-h sidebar-menu";
        menu.onclick = (e) => {
          e.stopPropagation();
          showChatOptionsMenu(s.id, s.title, menu);
        };

        item.appendChild(title);
        item.appendChild(menu);
        sidebarChat.appendChild(item);
      });
    },
  };

  // HIỂN THỊ TIN NHẮN
  function createMessage(text, sender) {
    const msg = document.createElement("div");
    msg.className = `message ${sender}-message show`;
    msg.textContent = text;
    chatContainer.appendChild(msg);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return msg;
  }

  function typeEffect(element, text, speed = 25) {
    let i = 0;
    const interval = setInterval(() => {
      element.textContent += text[i];
      i++;
      chatContainer.scrollTop = chatContainer.scrollHeight;
      if (i >= text.length) clearInterval(interval);
    }, speed);
  }

  // GỬI TIN NHẮN
  async function sendMessage() {
    const message = input.value.trim();
    if (!message) return;

    content.classList.add("chat-active");
    createMessage(message, "user");

    if (!currentSessionId) {
      currentSessionId = ChatHistory.createSession(message);
    }

    ChatHistory.addMessageToSession(currentSessionId, "user", message);
    input.value = "";

    const botMsg = createMessage("Bạn đợi mình chút xíu nha ", "bot");
    const dots = document.createElement("span");
    dots.className = "typing-dots";
    botMsg.appendChild(dots);
    let dotCount = 0;
    const typingAnim = setInterval(() => {
      dotCount = (dotCount + 1) % 4;
      dots.textContent = ".".repeat(dotCount);
    }, 400);

    try {
      const res = await fetch("http://127.0.0.1:5000/chat_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();
      clearInterval(typingAnim);
      botMsg.textContent = "";
      const reply = data.reply || "Xin lỗi, mình chưa hiểu ý bạn lắm.";
      typeEffect(botMsg, reply, 25);
      ChatHistory.addMessageToSession(currentSessionId, "bot", reply);
    } catch (err) {
      clearInterval(typingAnim);
      botMsg.textContent = "Mình bị mất kết nối một chút rồi.";
      console.error(err);
    }
  }

  // TẢI CHAT CŨ
  function loadChatSession(sessionId) {
    const sessions = ChatHistory.getAll();
    const session = sessions.find((s) => s.id === sessionId);
    if (!session) return;
    chatContainer.innerHTML = "";
    session.messages.forEach((msg) => createMessage(msg.text, msg.sender));
    currentSessionId = sessionId;
    content.classList.add("chat-active");
  }

  // MENU TUỲ CHỌN
  function showChatOptionsMenu(sessionId, sessionTitle, anchor) {
    const existing = document.querySelector(".chat-options-popup");
    if (existing) existing.remove();

    const menu = document.createElement("div");
    menu.className = "chat-options-popup";
    menu.innerHTML = `
      <button class="rename-chat">✏️ Đổi tên đoạn chat</button>
      <button class="delete-chat">🗑️ Xóa đoạn chat</button>
    `;
    document.body.appendChild(menu);

    const rect = anchor.getBoundingClientRect();
    menu.style.top = rect.bottom + "px";
    menu.style.left = rect.left + "px";

    menu.querySelector(".rename-chat").onclick = () => {
      menu.remove();
      showRenamePopup(sessionId, sessionTitle);
    };

    menu.querySelector(".delete-chat").onclick = () => {
      menu.remove();
      showDeleteConfirm(sessionId, sessionTitle);
    };

    document.addEventListener(
      "click",
      (e) => {
        if (!menu.contains(e.target) && e.target !== anchor) {
          menu.remove();
        }
      },
      { once: true },
    );
  }

  // POPUP ĐỔI TÊN
  function showRenamePopup(sessionId, oldTitle) {
    const overlay = document.createElement("div");
    overlay.className = "popup-overlay";
    overlay.innerHTML = `
      <div class="popup">
        <h3>Đổi tên đoạn chat</h3>
        <input type="text" id="new-chat-name" value="${oldTitle}">
        <div class="popup-actions">
          <button id="cancel-rename">Hủy</button>
          <button id="confirm-rename">Lưu</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    overlay.querySelector("#cancel-rename").onclick = () => overlay.remove();
    overlay.querySelector("#confirm-rename").onclick = () => {
      const newName = overlay.querySelector("#new-chat-name").value.trim();
      if (newName) ChatHistory.renameSession(sessionId, newName);
      overlay.remove();
    };
  }

  // POPUP XÓA CHAT
  function showDeleteConfirm(sessionId, title) {
    const overlay = document.createElement("div");
    overlay.className = "popup-overlay";
    overlay.innerHTML = `
      <div class="popup">
        <h3>Xóa đoạn chat</h3>
        <p>Bạn có chắc chắn muốn xóa đoạn chat "<b>${title}</b>" không?</p>
        <div class="popup-actions">
          <button id="cancel-delete">Hủy</button>
          <button id="confirm-delete">Xóa</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    overlay.querySelector("#cancel-delete").onclick = () => overlay.remove();
    overlay.querySelector("#confirm-delete").onclick = () => {
      ChatHistory.deleteSession(sessionId);
      if (currentSessionId === sessionId) {
        chatContainer.innerHTML = "";
        currentSessionId = null;
      }
      overlay.remove();
    };
  }

  // SỰ KIỆN
  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  });

  // Hover hiển thị tooltip
  toggleBtn.addEventListener("mouseenter", () => {
    if (isCollapsed) {
      moTooltip.style.opacity = "1";
      dongTooltip.style.opacity = "0";
    } else {
      dongTooltip.style.opacity = "1";
      moTooltip.style.opacity = "0";
    }
  });

  toggleBtn.addEventListener("mouseleave", () => {
    moTooltip.style.opacity = "0";
    dongTooltip.style.opacity = "0";
  });

  // Đóng / Mở sidebar
  toggleBtn.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
    isCollapsed = !isCollapsed;

    if (isCollapsed) {
      // Khi đóng → ẩn toàn bộ sidebar-chat
      sidebarChat.style.display = "none";
      moTooltip.style.opacity = "1";
      dongTooltip.style.opacity = "0";
    } else {
      // Khi mở → hiển thị lại
      sidebarChat.style.display = "block";
      dongTooltip.style.opacity = "1";
      moTooltip.style.opacity = "0";
    }

    // Ẩn tooltip sau 1.5s
    setTimeout(() => {
      moTooltip.style.opacity = "0";
      dongTooltip.style.opacity = "0";
    }, 1500);
  });

  function truncateText(text, maxLength = 20) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
  }
  // KHỞI TẠO
  ChatHistory.renderToSidebar();
});
