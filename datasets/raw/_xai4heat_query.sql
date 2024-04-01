

-- Kreiran je nalog za pristup bazi, lozinka je 'XAI4heat_OPC'.
-- Proverio sam port i IP adresu: U Firewall je otvoren TCP port 1433. LAN IP: 192.168.1.232, WAN IP: 160.99.21.189, administrator treba da podesi na ruteru port forward TCP1433 ka ovom serveru i proveri da li je LAN staticka ili da je gore navedena LAN IP adresa rezervisana za MAC adresu serverske mrezne kartice. Pozdrav.

-- Novi SQL upit za XAI4HEAT koji koriguje decimalnu tacku i negativne vrednosti
SELECT rpIstorijatTagova.DatumVremePromene,
    CASE WHEN kpTagovi.TipPodataka = 1 AND rpIstorijatTagova.Vrednost > 32768
        THEN (rpIstorijatTagova.Vrednost - 65535)/POWER(10.0, kpTagovi.DecimalnaTacka)
        ELSE rpIstorijatTagova.Vrednost/POWER(10.0, kpTagovi.DecimalnaTacka)
    END AS Vrednost,
    rpIstorijatTagova.Stanje,
    kpTagovi.*,
    kpUredjaji.Naziv AS Lokacija
FROM rpIstorijatTagova
JOIN kpTagovi ON rpIstorijatTagova.IdTaga = kpTagovi.IdTaga
JOIN kpUredjaji ON kpTagovi.IdUredjaja = kpUredjaji.IdUredjaja
WHERE kpUredjaji.Naziv IN ('TPS Lamela L4', 'TPS Lamela L8', 'TPS Lamela L12', 'TPS Lamela L17', 'TPS Lamela L22')
    AND DATEDIFF(day, DatumVremePromene, GETDATE()) <= 3000
ORDER BY DatumVremePromene DESC


-- Stari SQL upit
SELECT rpIstorijatTagova.*, kpTagovi.*, kpUredjaji.Naziv
FROM rpIstorijatTagova
JOIN kpTagovi ON rpIstorijatTagova.IdTaga = kpTagovi.IdTaga
JOIN kpUredjaji ON kpTagovi.IdUredjaja = kpUredjaji.IdUredjaja
WHERE kpUredjaji.Naziv IN ('TPS Lamela L4', 'TPS Lamela L8', 'TPS Lamela L12', 'TPS Lamela L17', 'TPS Lamela L22') 
AND DATEDIFF(day, DatumVremePromene, GETDATE()) <= 31
ORDER BY DatumVremePromene DESC
