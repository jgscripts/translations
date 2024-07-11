Locales = Locales or {} -- Translated by: PSYCHO#0281 (psycho0281)

Locales['hu'] = {	-- Erre a sorra is figyelj oda!!! Ez nem en.lua, hanem hu.lua, ezt fxmanifest, vagy resource lua-ban is említeni kell :)
  yes = "Igen",
  no = "Nem",
  garage = "Garázs",
  jobGarage = "Munka garázs",
  gangGarage = "Banda garázs",
  player = "Játékos",
  impound = "Lefoglalt",
  inGarage = "Garázsban",
  notInGarage = "Nincs a garázsban",
  car = "Autó",
  air = "Repülő",
  sea = "Hajó",
  fuel = "Üzemanyag",
  engine = "Motor",
  body = "Karosszéria",
  day = "nap",
  days = "napok",
  hour = "óra",
  hours = "órák",

  -- User Interface
  noVehicles = "Ebben a garázsban nincsenek járművek",
  vehicles = "jármű(vek)",
  vehiclePlate = "Rendszám",
  vehicleNotInGarage = "A jármű nincs garázsban",
  vehicleTakeOut = "Vezetés",
  vehicleReturnAndTakeOut = "Vissza és vezetés",
  vehicleReturnToOwnersGarage = "Visszarakás a tulajdonos garázsába",
  transferToGarageOrPlayer = "Átszállítás egy garázsba vagy a játékosnak",
  transferToGarage = "Átszállítás garázsba",
  transferToPlayer = "Átszállítás játékosnak",
  vehicleTransfer = "Átadás",
  noAvailableGarages = "Nincs szabad garázs",
  currentGarage = "Jelenlegi garázs",
  noPlayersOnline = "Nincs ilyen elérhető játékos",
  createPrivateGarage = "Privát garázs készítése",
  pgAlertHeadsUp = "Fel a fejjel!",
  pgAlertText = "A garázs létrejön, és a járművek pontosan azon a helyen és irányban fognak megjelenni, ahol éppen áll.",
  garageName = "Garázs neve",
  impoundInformation = "Lefoglaltak információ",
  impoundedBy = "Lefoglalta",
  impoundedReason = "Indok",
  impoundPlayerCanCollect = "Gépkocsiját átveheti a lefoglaltakban.",
  impoundCollectionContact = "Kérjük, vegye fel a kapcsolatot %{value} a jármű átvétele érdekében.",
  impoundNoVehicles = "Nincs lefoglalt autó",
  impoundAvailable = "Elérhető",
  impoundRetrievableByOwner = "Tulajdonos által átvehető",
  impoundNoReason = "Indoklás nélkül",
  impoundVehicle = "Lefoglalt jármű",
  impoundReasonField = "Indok (nem kötelező)",
  impoundTime = "Lefoglalt idő",
  impoundAvailableImmediately = "Azonnal elérhető",
  impoundCost = "Költség",
  changeVehiclePlate = "Rendszámtábla cserélése",
  newPlate = "Új rendszám",
  search = "Keresés név vagy rendszám alapján",
  noPrivateGarages = "Nincs privát garázsod",
  editPrivateGarage = "Privát garázs szerkesztése",
  owners = "Tulajdonos(ok)",
  location = "Elhelyezkedés",
  next = "Következő",
  previous = "Előző",
  page = "Oldal",
  of = "nak,-nek",
  show = "Mutatás",
  save = "Mentés",
  edit = "Szerkesztés",
  delete = "Törlés",
  garageDeleteConfirm = "Biztosan törli ezt a garázst?",
  privGarageSearch = "Keresés név alapján",
  garageUpdatedSuccess = "Garázs sikeresen frissítve!",
  getCurrentCoords = "Jelenlegi koordináták lekérdezése",
  identifier = "Azonosító",
  name = "Név",
  noPlayers = "Nincsenek hozzáadott játékosok",
  addPlayer = "Játékos hozzáadása",
  loadingVehicle = "Jármű betöltése...",
  vehicleSetup = "Járműbeállítás",
  extras = "Extrák",
  extra = "Extra",
  liveries = "Festések",
  livery = "Festés",
  noLiveries = "Nincs elérhető festés",
  noExtras = "Nincsenek extrák",
  none = "Semmi",
  vehicleNeedsService = "Needs Service",
  type = "Type",

  -- Notifications
  insertVehicleTypeError = "Ebben a garázsban csak %{value} járműtípusokat tárolhat",
  insertVehiclePublicError = "Nem tárolhat munkahelyi vagy bandajárműveket nyilvános garázsokban",
  vehicleParkedSuccess = "Jármű leparkolva",
  vehicleNotOwnedError = "Nem Ön a tulajdonosa ennek a járműnek",
  vehicleNotOwnedByPlayerError = "A jármű nem játékos tulajdonában van",
  notEnoughMoneyError = "Nincs elég pénzed a bankodban",
  vehicleNotYoursError = "A jármű nem az Öné",
  notJobOrGangVehicle = "Ez nem %{value} jármű",
  invalidGangError = "Nem adott meg érvényes bandát",
  invalidJobError = "Nem adott meg érvényes állást",
  notInsideVehicleError = "Nem ülsz járműben",
  vehicleAddedToGangGarageSuccess = "A jármű bekerült a %{value} banda garázsba!",
  vehicleAddedToJobGarageSuccess = "A jármű bekerült a %{value} munka garázsba!",
  moveCloserToVehicleError = "Közelebb kell mennie a járműhöz",
  noVehiclesNearbyError = "A közelben nincsenek járművek",
  commandPermissionsError = "Nem használhatja ezt a parancsot",
  actionNotAllowedError = "Ehhez nincs engedélye",
  garageNameExistsError = "A garázsnév már létezik",
  vehiclePlateExistsError = "A jármű táblája már használatban van",
  playerNotOnlineError = "Játékos nincs online",
  vehicleTransferSuccess = "Jármű átadva neki: %{value}",
  vehicleTransferSuccessGeneral = "A jármű sikeresen átadva",
  vehicleReceived = "Te kaptál egy járművet, ezzel a rendszámmal %{value}",
  vehicleImpoundSuccess = "Sikeresen lefoglalták a járművet",
  vehicleImpoundRemoveSuccess = "Sikeresen kilett véve a lefoglaltakból",
  vehicleImpoundReturnedToOwnerSuccess = "A jármű visszakerült a tulajdonos garázsába",
  garageCreatedSuccess = "Garázs sikeresen létrehozva!",
  vehiclePlateUpdateSuccess = "Rendszám beállítva erre: %{value}",
  vehicleDeletedSuccess = "Jármű törölve az adatbázisból %{value}",
  playerIsDead = "Ezt nem teheted, amíg halott vagy",

  -- Commands
  cmdSetGangVehicle = "Jelenlegi jármű hozzáadása a banda garázshoz",
  cmdRemoveGangVehicle = "Banda garázsból visszarakás a játékos tulajdonába",
  cmdSetJobVehicle = "Jelenlegi jármű hozzáadása a munka garázshoz",
  cmdRemoveJobVehicle = "Munka garázsból visszarakás a játékos tulajdonába",
  cmdArgGangName = "Banda neve",
  cmdArgJobName = "Mubka neve",
  cmgArgMinGangRank = "Minimum banda rang",
  cmgArgMinJobRank = "Minimum beosztás",
  cmdArgPlayerId = "Az új tulajdonos játékosazonosítója",
  cmdImpoundVehicle = "Jármű lefoglalása",
  cmdChangePlate = "Rendszámtábla cseréje (csak rendszergazda)",
  cmdDeleteVeh = "Jármű törlése az adatbázisból (csak rendszergazda)",
  cmdCreatePrivGarage = "Privát garázs létrehozása egy játékos számára",

  -- v3
  vehicleStoreError = "You cannot store this vehicle here",
  mins = "mins",
  noVehiclesAvailableToDrive = "There are no vehicles available to drive",
}