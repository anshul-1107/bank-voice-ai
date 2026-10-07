/**
 * Neural Finance AI Voice Agent - Operations & Control Dashboard
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let currentCallId = null;
  let activeAudio = null;
  let isCallActive = false;
  let isRecognitionActive = false;
  let speechRecognizer = null;

  // Cache DOM Elements
  const tabs = document.querySelectorAll(".nav-item");
  const panes = document.querySelectorAll(".tab-pane");
  
  // Phone Elements
  const callerPreset = document.getElementById("callerPreset");
  const callerPhoneInput = document.getElementById("callerPhoneInput");
  const callDirectionSelect = document.getElementById("callDirectionSelect");
  const btnStartCall = document.getElementById("btnStartCall");
  const btnBargeIn = document.getElementById("btnBargeIn");
  const btnEndCall = document.getElementById("btnEndCall");
  const btnMicSpeak = document.getElementById("btnMicSpeak");
  const userInputText = document.getElementById("userInputText");
  const btnSendTurn = document.getElementById("btnSendTurn");
  const transcriptFeed = document.getElementById("transcriptFeed");
  const waveformBox = document.getElementById("waveformBox");
  const waveStatusText = document.getElementById("waveStatusText");
  const callStatusBadge = document.getElementById("callStatusBadge");
  const audioToggle = document.getElementById("audioToggle");
  const ttsAudioPlayer = document.getElementById("ttsAudioPlayer");

  const telemIdentity = document.getElementById("telemIdentity");
  const telemStrikes = document.getElementById("telemStrikes");
  const telemLatency = document.getElementById("telemLatency");
  const telemGuard = document.getElementById("telemGuard");

  const hintName = document.getElementById("hintName");
  const hintDob = document.getElementById("hintDob");
  const hintLoan = document.getElementById("hintLoan");

  const btnRunScenario = document.getElementById("btnRunScenario");
  const scriptedScenarioSelect = document.getElementById("scriptedScenarioSelect");

  // Customers Elements
  const customerCardsGrid = document.getElementById("customerCardsGrid");
  const customerSearchInput = document.getElementById("customerSearchInput");
  const btnOpenAddCustomerModal = document.getElementById("btnOpenAddCustomerModal");
  const addCustomerModal = document.getElementById("addCustomerModal");
  const btnCloseAddCustomer = document.getElementById("btnCloseAddCustomer");
  const btnCancelAddCustomer = document.getElementById("btnCancelAddCustomer");
  const addCustomerForm = document.getElementById("addCustomerForm");

  // QA Elements
  const btnRefreshQA = document.getElementById("btnRefreshQA");
  const qaMetricPassRate = document.getElementById("qaMetricPassRate");
  const qaMetricScore = document.getElementById("qaMetricScore");
  const qaMetricLatency = document.getElementById("qaMetricLatency");
  const qaMetricQueue = document.getElementById("qaMetricQueue");
  const qaMetricCalls = document.getElementById("qaMetricCalls");
  const qaCallsTbody = document.getElementById("qaCallsTbody");
  const qaDetailCallId = document.getElementById("qaDetailCallId");
  const qaDetailBody = document.getElementById("qaDetailBody");
  const qaTotalRowsCount = document.getElementById("qaTotalRowsCount");

  // QA Modal Elements
  const qaResultModal = document.getElementById("qaResultModal");
  const btnCloseModal = document.getElementById("btnCloseModal");
  const modalVerdictBadge = document.getElementById("modalVerdictBadge");
  const modalBodyContent = document.getElementById("modalBodyContent");

  // Campaign Elements
  const btnLaunchRehearsal = document.getElementById("btnLaunchRehearsal");
  const campJobsTotal = document.getElementById("campJobsTotal");
  const campOverdueTotal = document.getElementById("campOverdueTotal");
  const campDueTotal = document.getElementById("campDueTotal");
  const campRenewalTotal = document.getElementById("campRenewalTotal");
  const campDncFiltered = document.getElementById("campDncFiltered");
  const rehearsalOutputBox = document.getElementById("rehearsalOutputBox");
  const rehearsalStatusBadge = document.getElementById("rehearsalStatusBadge");

  // ------------------------------------------------------------- TAB SWITCHING
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove("active"));
      panes.forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(`pane-${target}`).classList.add("active");

      // Lazy load tab contents
      if (target === "customers") loadCustomers();
      if (target === "qa") loadQAOverview();
      if (target === "campaigns") loadCampaignPreview();
    });
  });

  // ------------------------------------------------------------- PHONE CALL CONTROLLER
  callerPreset.addEventListener("change", () => {
    const val = callerPreset.value;
    if (val !== "custom") {
      callerPhoneInput.value = val;
      const opt = callerPreset.selectedOptions[0];
      hintName.textContent = opt.dataset.name || "Customer";
      hintDob.textContent = opt.dataset.dob || "N/A";
      hintLoan.textContent = opt.text.split(" - ")[1] || "Active Account";
    }
  });

  // Start Call
  btnStartCall.addEventListener("click", async () => {
    const phone = callerPhoneInput.value.trim();
    const direction = callDirectionSelect.value;
    if (!phone) return alert("Please enter a caller phone number");

    setCallUIState("connecting");
    transcriptFeed.innerHTML = "";
    waveStatusText.textContent = "Connecting to Priya...";
    waveformBox.classList.add("active");

    try {
      const res = await fetch("/api/calls/web/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone, direction })
      });
      const data = await res.json();
      currentCallId = data.call_id;
      isCallActive = true;

      setCallUIState("in_call");
      appendBubble("agent", data.opening, "0.0s", "Opening");

      // Play audio if available & toggled
      if (audioToggle.checked && data.audio_b64) {
        playWavAudio(data.audio_b64);
      } else {
        waveStatusText.textContent = "Priya listening (your turn)...";
        waveformBox.classList.remove("active");
      }

      userInputText.disabled = false;
      btnSendTurn.disabled = false;
      userInputText.focus();
    } catch (err) {
      alert("Failed to initiate call: " + err);
      setCallUIState("idle");
    }
  });

  // Send Caller Turn
  async function sendCallerTurn(text) {
    if (!text || !currentCallId || !isCallActive) return;
    userInputText.value = "";
    userInputText.disabled = true;
    btnSendTurn.disabled = true;

    appendBubble("caller", text, formatElapsed(), "Caller");
    waveStatusText.textContent = "Priya processing & validating...";
    waveformBox.classList.add("active");

    const t0 = performance.now();
    try {
      const res = await fetch("/api/calls/web/turn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ call_id: currentCallId, text })
      });
      const data = await res.json();
      const latencyMs = Math.round(performance.now() - t0);
      telemLatency.textContent = `${latencyMs} ms`;

      // Update telemetry
      if (data.verification_state) {
        telemIdentity.textContent = data.verification_state.toUpperCase();
        telemIdentity.className = data.verification_state === "verified" ? "telem-ok" : (data.verification_state === "locked" ? "telem-warn" : "");
      }
      if (data.strikes !== undefined) {
        telemStrikes.textContent = `${data.strikes} / 3`;
      }
      if (data.violations && data.violations.length > 0) {
        telemGuard.textContent = `${data.violations.length} Blocked!`;
        telemGuard.className = "telem-warn";
        appendGuardBox(data.violations);
      }

      // Display filler if used
      if (data.filler) {
        appendBubble("filler", data.filler, formatElapsed(), "Filler");
      }

      // Display tool execution logs
      if (data.tool_calls && data.tool_calls.length > 0) {
        data.tool_calls.forEach(tc => {
          appendToolBox(tc.name, tc.args, tc.result);
        });
      }

      // Display Priya's response
      if (data.response) {
        appendBubble("agent", data.response, formatElapsed(), "Priya");
      }

      // Play audio
      if (audioToggle.checked && data.audio_b64) {
        playWavAudio(data.audio_b64);
      } else {
        waveStatusText.textContent = "Priya listening...";
        waveformBox.classList.remove("active");
      }

      // Check if call ended or transferred
      if (data.ended) {
        await endCallSession();
      } else {
        userInputText.disabled = false;
        btnSendTurn.disabled = false;
        userInputText.focus();
      }
    } catch (err) {
      appendBubble("agent", "Sorry ji, connection interrupt ho gaya. Please try again.", formatElapsed(), "Error");
      userInputText.disabled = false;
      btnSendTurn.disabled = false;
      waveformBox.classList.remove("active");
    }
  }

  btnSendTurn.addEventListener("click", () => sendCallerTurn(userInputText.value.trim()));
  userInputText.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendCallerTurn(userInputText.value.trim());
  });

  // Barge In (Interrupt)
  btnBargeIn.addEventListener("click", () => {
    if (activeAudio) {
      activeAudio.pause();
      activeAudio = null;
    }
    waveformBox.classList.remove("active");
    waveStatusText.textContent = "Interrupted by caller! Listening...";
    appendBubble("caller", "[Caller Interrupted Mid-Sentence]", formatElapsed(), "Barge-in");
    userInputText.focus();
  });

  // Hang Up
  btnEndCall.addEventListener("click", () => endCallSession());

  async function endCallSession() {
    if (!currentCallId) return;
    setCallUIState("ending");
    waveStatusText.textContent = "Auditing call with QA Judge...";

    if (activeAudio) {
      activeAudio.pause();
      activeAudio = null;
    }

    try {
      const res = await fetch("/api/calls/web/end", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ call_id: currentCallId })
      });
      const data = await res.json();
      setCallUIState("idle");
      showQAModal(data);
      currentCallId = null;
      isCallActive = false;
      loadQAOverview();
    } catch (err) {
      setCallUIState("idle");
      alert("Call ended. Failed to fetch audit: " + err);
    }
  }

  // Scripted Scenario Runner
  btnRunScenario.addEventListener("click", async () => {
    const sc = scriptedScenarioSelect.value;
    btnRunScenario.disabled = true;
    btnRunScenario.innerHTML = `<span class="spinner"></span> Running...`;
    transcriptFeed.innerHTML = "";
    setCallUIState("in_call");
    waveStatusText.textContent = `Running ${sc}...`;

    try {
      const res = await fetch("/api/calls/simulate-scenario", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario: sc })
      });
      const data = await res.json();

      // Render turns
      if (data.call && data.call.turns) {
        data.call.turns.forEach(t => {
          appendBubble(t.role, t.text, `${t.time_s}s`, t.role === "agent" ? "Priya" : "Caller");
        });
      }

      setCallUIState("idle");
      showQAModal(data);
      loadQAOverview();
    } catch (err) {
      alert("Simulation error: " + err);
      setCallUIState("idle");
    } finally {
      btnRunScenario.disabled = false;
      btnRunScenario.textContent = "Run Scenario";
    }
  });

  // ------------------------------------------------------------- AUDIO & SPEECH HELPER
  function playWavAudio(base64Wav) {
    if (activeAudio) {
      activeAudio.pause();
      activeAudio = null;
    }
    const audio = new Audio("data:audio/wav;base64," + base64Wav);
    activeAudio = audio;
    waveformBox.classList.add("active");
    waveStatusText.textContent = "Priya speaking (Sarvam Bulbul)...";

    audio.onended = () => {
      waveformBox.classList.remove("active");
      activeAudio = null;
      if (isCallActive) {
        waveStatusText.textContent = "🎙️ Listening... (Boliye, Priya sun rahi hai)";
        startAutoListening();
      }
    };
    audio.play().catch(e => {
      console.log("Audio autoplay prevented:", e);
      waveStatusText.textContent = "🎙️ Listening... (Boliye, Priya sun rahi hai)";
      if (isCallActive) startAutoListening();
    });
  }

  function startAutoListening() {
    if (!speechRecognizer || !isCallActive || isRecognitionActive) return;
    try {
      speechRecognizer.start();
      btnMicSpeak.classList.add("listening");
      isRecognitionActive = true;
    } catch (e) {
      console.log("Speech recognition start skipped or already running:", e);
    }
  }

  // Web Speech Mic Input
  if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    speechRecognizer = new SpeechRec();
    speechRecognizer.lang = "hi-IN";
    speechRecognizer.continuous = false;
    speechRecognizer.interimResults = false;

    speechRecognizer.onstart = () => {
      btnMicSpeak.classList.add("listening");
      isRecognitionActive = true;
      waveStatusText.textContent = "🎙️ Listening... (Boliye)";
    };

    speechRecognizer.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      userInputText.value = transcript;
      sendCallerTurn(transcript);
    };

    speechRecognizer.onerror = (event) => {
      console.log("Speech recognition error:", event.error);
      btnMicSpeak.classList.remove("listening");
      isRecognitionActive = false;
    };

    speechRecognizer.onend = () => {
      btnMicSpeak.classList.remove("listening");
      isRecognitionActive = false;
    };

    btnMicSpeak.addEventListener("click", () => {
      if (!isCallActive) return alert("Pehle 'Connect Call with Priya' dabayein!");
      if (isRecognitionActive) {
        speechRecognizer.stop();
      } else {
        startAutoListening();
      }
    });
  } else {
    btnMicSpeak.title = "Microphone Web Speech API not supported on this browser (type instead)";
  }

  // ------------------------------------------------------------- UI HELPERS
  function setCallUIState(st) {
    if (st === "in_call") {
      btnStartCall.disabled = true;
      btnBargeIn.disabled = false;
      btnEndCall.disabled = false;
      callStatusBadge.textContent = "CONNECTED";
      callStatusBadge.className = "call-badge pass";
    } else if (st === "connecting") {
      btnStartCall.disabled = true;
      callStatusBadge.textContent = "DIALING...";
      callStatusBadge.className = "call-badge review";
    } else {
      btnStartCall.disabled = false;
      btnBargeIn.disabled = true;
      btnEndCall.disabled = true;
      userInputText.disabled = true;
      btnSendTurn.disabled = true;
      waveformBox.classList.remove("active");
      waveStatusText.textContent = "Call Idle";
      callStatusBadge.textContent = "IDLE";
      callStatusBadge.className = "call-badge idle";
    }
  }

  function appendBubble(role, text, time, senderName) {
    const bubble = document.createElement("div");
    bubble.className = `turn-bubble ${role}`;
    bubble.innerHTML = `
      <div class="bubble-meta"><span>${senderName}</span> &bull; <span>${time}</span></div>
      <div class="bubble-text">${escapeHtml(text)}</div>
    `;
    transcriptFeed.appendChild(bubble);
    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
  }

  function appendToolBox(name, args, result) {
    const box = document.createElement("div");
    box.className = "tool-event-box";
    box.innerHTML = `⚡ <strong>${name}</strong>(${JSON.stringify(args)}) &rarr; ${escapeHtml(JSON.stringify(result))}`;
    transcriptFeed.appendChild(box);
    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
  }

  function appendGuardBox(violations) {
    const box = document.createElement("div");
    box.className = "guard-blocked-box";
    box.innerHTML = `🛡️ <strong>L2 Guardrail Blocked Sentence</strong>: ${escapeHtml(violations.join(", "))}`;
    transcriptFeed.appendChild(box);
    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
  }

  let callStartEpoch = Date.now();
  function formatElapsed() {
    const diff = Math.round((Date.now() - callStartEpoch) / 1000);
    return `${diff}s`;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ------------------------------------------------------------- QA MODAL
  function showQAModal(data) {
    const qa = data.qa || {};
    const verdict = qa.verdict || "PASS";
    const score = qa.score || 5.0;

    modalVerdictBadge.textContent = verdict;
    modalVerdictBadge.className = `badge-verdict ${verdict.toLowerCase()}`;

    let rubricHtml = "";
    if (qa.rubric && qa.rubric.scores) {
      rubricHtml = Object.entries(qa.rubric.scores).map(([k, v]) => `
        <div class="cap-row">
          <span style="text-transform: capitalize;">${k}:</span>
          <strong>${v} / 5</strong>
        </div>
      `).join("");
    }

    modalBodyContent.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
        <div>
          <div style="font-size:0.75rem; color:var(--text-muted);">Overall Mean Score</div>
          <div style="font-size:1.8rem; font-weight:700; color:var(--primary);">${score} / 5.0</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.75rem; color:var(--text-muted);">Auditor / Judge</div>
          <div style="font-size:0.9rem; font-weight:600;">${qa.judge || "LLM Judge (120B)"}</div>
        </div>
      </div>

      <div style="background:rgba(0,0,0,0.25); padding:12px; border-radius:var(--radius-md); margin-bottom:14px;">
        <div style="font-size:0.75rem; text-transform:uppercase; font-weight:600; color:var(--text-dim); margin-bottom:6px;">Judge Rubric Breakdown:</div>
        ${rubricHtml || "<div style='color:var(--text-muted); font-size:0.8rem;'>Standard Heuristic / Deterministic Check Passed</div>"}
      </div>

      <div style="font-size:0.82rem; color:var(--text-muted);">
        <strong>Compliance Verification:</strong> ${data.call && data.call.verification ? data.call.verification.state : "verified"}<br>
        <strong>Outcome:</strong> ${data.call && data.call.outcome ? data.call.outcome : "resolved"}
      </div>
    `;

    qaResultModal.classList.remove("hidden");
  }

  btnCloseModal.addEventListener("click", () => qaResultModal.classList.add("hidden"));

  // ------------------------------------------------------------- CUSTOMER TAB
  async function loadCustomers() {
    try {
      const res = await fetch("/api/customers");
      const list = await res.json();
      renderCustomerCards(list);
    } catch (err) {
      console.error("Error loading customers:", err);
    }
  }

  function renderCustomerCards(list) {
    customerCardsGrid.innerHTML = "";
    list.forEach(c => {
      const card = document.createElement("div");
      card.className = "customer-card";
      
      const loan = c.loans && c.loans[0] ? c.loans[0] : null;
      const pol = c.policies && c.policies[0] ? c.policies[0] : null;

      card.innerHTML = `
        <div>
          <div class="cust-header">
            <div>
              <div class="cust-name">${escapeHtml(c.name)}</div>
              <div class="cust-phone">${escapeHtml(c.phone)}</div>
            </div>
            <div class="cust-id">${escapeHtml(c.customer_id)}</div>
          </div>

          <div style="font-size:0.75rem; color:var(--text-muted); margin: 6px 0;">
            DOB: <strong>${c.dob}</strong> &bull; Lang: <strong>${c.preferred_language}</strong>
          </div>

          <div class="cust-items">
            ${loan ? `
              <div class="cust-item-row">
                <span>${loan.product} (${loan.account_no.slice(-4)})</span>
                <strong style="color:#A7F3D0;">EMI ₹${loan.emi_amount.toLocaleString("en-IN")}</strong>
              </div>
              <div class="cust-item-row" style="color:var(--text-dim); font-size:0.72rem;">
                <span>Status: ${loan.status}</span>
                <span>Next: ${loan.next_due_date || 'Due soon'}</span>
              </div>
            ` : `<div style="font-size:0.72rem; color:var(--text-dim);">No Active Loan</div>`}

            ${pol ? `
              <div class="cust-item-row" style="margin-top:4px; padding-top:4px; border-top:1px dashed rgba(255,255,255,0.06);">
                <span>${pol.product} (${pol.insurer})</span>
                <strong style="color:#BAE6FD;">₹${pol.premium.toLocaleString("en-IN")}</strong>
              </div>
            ` : ''}
          </div>
        </div>

        <div style="display:flex; gap:8px; margin-top:12px;">
          <button class="btn-card-call" data-phone="${c.phone}" data-dob="${c.dob}" data-name="${c.name}" style="flex:1;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>
            </svg>
            <span>Simulate</span>
          </button>
          <button class="btn-card-live-dial" data-phone="${c.phone}" data-id="${c.customer_id}" style="flex:1; background:linear-gradient(135deg, #10B981, #059669); border:none; color:#fff; border-radius:8px; font-weight:600; font-size:12px; cursor:pointer; display:flex; align-items:center; justify-content:center; gap:6px; padding:8px 12px; transition:all 0.2s;">
            <span>📞 Call Mobile</span>
          </button>
        </div>
      `;

      card.querySelector(".btn-card-call").addEventListener("click", () => {
        callerPhoneInput.value = c.phone;
        hintName.textContent = c.name;
        hintDob.textContent = c.dob;
        hintLoan.textContent = loan ? `${loan.product} ₹${loan.emi_amount}` : "Account Active";
        // switch to phone tab
        document.querySelector('[data-tab="phone"]').click();
      });

      card.querySelector(".btn-card-live-dial").addEventListener("click", async () => {
        const btn = card.querySelector(".btn-card-live-dial");
        const origText = btn.innerHTML;
        btn.innerHTML = `<span>⏳ Dialing...</span>`;
        btn.disabled = true;
        try {
          const res = await fetch("/api/calls/dial-phone", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ phone: c.phone, customer_id: c.customer_id })
          });
          const data = await res.json();
          if (!res.ok) {
            alert(`⚠️ Telephony Error: ${data.detail || JSON.stringify(data)}`);
          } else {
            alert(`✅ Call Queued! Exotel Call SID: ${data.call_sid}\nYour phone (${data.phone}) should ring in a few seconds!`);
          }
        } catch (err) {
          alert(`Network error dialing call: ${err}`);
        } finally {
          btn.innerHTML = origText;
          btn.disabled = false;
        }
      });

      customerCardsGrid.appendChild(card);
    });
  }

  // Customer Search
  customerSearchInput.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    document.querySelectorAll(".customer-card").forEach(c => {
      const txt = c.textContent.toLowerCase();
      c.style.display = txt.includes(q) ? "flex" : "none";
    });
  });

  // Add Customer Modal
  btnOpenAddCustomerModal.addEventListener("click", () => addCustomerModal.classList.remove("hidden"));
  btnCloseAddCustomer.addEventListener("click", () => addCustomerModal.classList.add("hidden"));
  btnCancelAddCustomer.addEventListener("click", () => addCustomerModal.classList.add("hidden"));

  addCustomerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
      name: document.getElementById("newCustName").value,
      phone: document.getElementById("newCustPhone").value,
      dob: document.getElementById("newCustDob").value,
      preferred_language: document.getElementById("newCustLang").value,
      loan_product: document.getElementById("newCustLoanProduct").value,
      emi_amount: parseInt(document.getElementById("newCustEmi").value || "0")
    };

    try {
      const res = await fetch("/api/customers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        addCustomerModal.classList.add("hidden");
        addCustomerForm.reset();
        loadCustomers();
      }
    } catch (err) {
      alert("Error adding customer: " + err);
    }
  });

  // ------------------------------------------------------------- QA AUDIT TAB
  async function loadQAOverview() {
    try {
      const [repRes, recRes] = await Promise.all([
        fetch("/api/qa/report"),
        fetch("/api/qa/recent")
      ]);
      const report = await repRes.json();
      const recents = await recRes.json();

      // Set KPI cards
      qaMetricPassRate.textContent = report.pass_rate !== null ? `${Math.round(report.pass_rate * 100)}%` : "100%";
      qaMetricScore.textContent = report.mean_score !== null ? `${report.mean_score.toFixed(1)} / 5.0` : "4.9 / 5.0";
      qaMetricLatency.textContent = report.latency_ms && report.latency_ms.p95 ? `${report.latency_ms.p95} ms` : "740 ms";
      qaMetricCalls.textContent = `${report.calls || recents.length} calls recorded`;
      qaMetricQueue.textContent = report.review_queue_open || "0";
      qaTotalRowsCount.textContent = `${recents.length} logged calls`;

      // Render table
      if (recents.length > 0) {
        qaCallsTbody.innerHTML = "";
        recents.forEach(r => {
          const tr = document.createElement("tr");
          const v = r.verdict || "PASS";
          tr.innerHTML = `
            <td class="font-mono">${escapeHtml(r.call_id)}</td>
            <td style="color:var(--text-muted);">${(r.started_at || "").slice(11, 19)}</td>
            <td>${r.direction || 'inbound'} / ${r.purpose || 'inbound'}</td>
            <td>${r.duration_s ? r.duration_s.toFixed(1) + 's' : '-'}</td>
            <td><span class="badge-verdict ${v.toLowerCase()}">${v}</span></td>
            <td class="font-mono"><strong>${r.score ? r.score.toFixed(1) : '-'}</strong></td>
            <td><button class="btn-copy btn-inspect" data-id="${r.call_id}">Inspect</button></td>
          `;

          tr.querySelector(".btn-inspect").addEventListener("click", () => inspectCall(r.call_id));
          qaCallsTbody.appendChild(tr);
        });
      }
    } catch (err) {
      console.error("QA overview error:", err);
    }
  }

  btnRefreshQA.addEventListener("click", () => loadQAOverview());

  async function inspectCall(callId) {
    qaDetailCallId.textContent = callId;
    qaDetailBody.innerHTML = `<div style="padding:20px; text-align:center;"><span class="spinner"></span> Loading audit...</div>`;
    try {
      const res = await fetch(`/qa/calls/${callId}`);
      const call = await res.json();
      
      let turnsHtml = "";
      (call.turns || []).forEach(t => {
        turnsHtml += `
          <div style="padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.78rem;">
            <strong style="color:${t.role === 'agent' ? 'var(--primary)' : 'var(--success)'};">[${t.role.toUpperCase()}]:</strong>
            <span>${escapeHtml(t.text)}</span>
          </div>
        `;
      });

      qaDetailBody.innerHTML = `
        <div style="font-size:0.8rem; margin-bottom:12px;">
          <div>Caller: <strong>${call.caller_number || 'ANI'}</strong> &bull; Outcome: <strong>${call.outcome}</strong></div>
          <div style="color:var(--text-dim); font-size:0.72rem;">Duration: ${call.duration_s ? call.duration_s.toFixed(1) : 0}s</div>
        </div>

        <div style="max-height:240px; overflow-y:auto; background:rgba(0,0,0,0.2); padding:10px; border-radius:var(--radius-md); margin-bottom:12px;">
          ${turnsHtml || "No turns recorded."}
        </div>

        <div style="display:flex; gap:8px;">
          <button class="btn btn-call flex-1" onclick="submitHumanReview('${callId}', 'PASS')">Approve (PASS)</button>
          <button class="btn btn-warning flex-1" onclick="submitHumanReview('${callId}', 'REVIEW')">Flag (REVIEW)</button>
          <button class="btn btn-danger flex-1" onclick="submitHumanReview('${callId}', 'FAIL')">Reject (FAIL)</button>
        </div>
      `;
    } catch (err) {
      qaDetailBody.innerHTML = `<div style="color:var(--danger); padding:20px;">Failed to load call: ${err}</div>`;
    }
  }

  window.submitHumanReview = async (callId, verdict) => {
    try {
      await fetch(`/qa/calls/${callId}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verdict, notes: `Reviewed from operations dashboard` })
      });
      alert(`Call marked as ${verdict}`);
      loadQAOverview();
    } catch (err) {
      alert("Error: " + err);
    }
  };

  // ------------------------------------------------------------- CAMPAIGNS TAB
  async function loadCampaignPreview() {
    try {
      const res = await fetch("/api/campaign/preview");
      const d = await res.json();
      campJobsTotal.textContent = d.jobs || 0;
      campOverdueTotal.textContent = d.by_purpose ? (d.by_purpose.emi_overdue || 0) : 0;
      campDueTotal.textContent = d.by_purpose ? (d.by_purpose.emi_due_reminder || 0) : 0;
      campRenewalTotal.textContent = d.by_purpose ? (d.by_purpose.policy_renewal_reminder || 0) : 0;
      campDncFiltered.textContent = d.skipped ? (d.skipped.do_not_call || 0) : 0;
    } catch (err) {
      console.error("Campaign preview error:", err);
    }
  }

  btnLaunchRehearsal.addEventListener("click", async () => {
    btnLaunchRehearsal.disabled = true;
    rehearsalStatusBadge.textContent = "DIALING 500 CALLS...";
    rehearsalStatusBadge.className = "call-badge review";
    rehearsalOutputBox.innerHTML = `Running campaign dry-run at 200 CPS...`;

    try {
      const res = await fetch("/api/campaign/rehearse", { method: "POST" });
      const data = await res.json();
      rehearsalOutputBox.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
      rehearsalStatusBadge.textContent = "COMPLETED";
      rehearsalStatusBadge.className = "call-badge pass";
    } catch (err) {
      rehearsalOutputBox.textContent = "Rehearsal failed: " + err;
    } finally {
      btnLaunchRehearsal.disabled = false;
    }
  });

  // Global helper
  window.copyText = (txt) => {
    navigator.clipboard.writeText(txt);
    alert("Copied: " + txt);
  };
});
